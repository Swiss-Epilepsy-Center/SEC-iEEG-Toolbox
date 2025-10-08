from __future__ import annotations
import re 
import numpy as np
import nibabel as nb 
from collections import defaultdict  

from .coords import read_subject_electrodes, extract_mni_slices
from .roi import load_pial_mesh, _fs_aseg_to_mesh_list, smooth_trimesh



def _require_plotly():
    try:
        import plotly.graph_objects as go
        return go
    except ImportError as e:
        raise ImportError("Install Plotly: pip install plotly") from e

def _pretty_elname(full: str) -> str:
    """
    Convert raw electrode identifier strings into nice labels.
    Examples:
      "113:1" -> "Electrode 1"
      "119:el 3" -> "Electrode 3"
      "5" -> "Electrode 5"
    """

    if ":" in full:
        el = full.split(":", 1)[1].strip()
    else:
        el = full.strip()

    import re
    m = re.match(r'(?i)(?:el\s*)?(\d+)$', el)
    if m:
        return m.group(1)
    else:
        return el

# ---------- helpers ----------

def pretty_roi_name(name: str) -> str:
    """
    Normalize ROI / fiber names:
      'Left-Hippocampus' -> 'Left Hippocampus'
      'BasalGanglia_l'   -> 'Left Basal Ganglia'
      'MLF_r'            -> 'Right MLF'
      'lh.Thalamus'      -> 'Left Thalamus'
    """
    n = str(name)

    # left/right prefixes or suffixes
    lr = None
    if re.match(r"^(lh\.|left[-_\s])", n, flags=re.I): lr, n = "Left",  re.sub(r"^(lh\.|left[-_\s])", "", n, flags=re.I)
    if re.match(r"^(rh\.|right[-_\s])", n, flags=re.I): lr, n = "Right", re.sub(r"^(rh\.|right[-_\s])", "", n, flags=re.I)
    if re.search(r"([-_\s]l|_l)$", n, flags=re.I): lr, n = "Left",  re.sub(r"([-_\s]l|_l)$", "", n, flags=re.I)
    if re.search(r"([-_\s]r|_r)$", n, flags=re.I): lr, n = "Right", re.sub(r"([-_\s]r|_r)$", "", n, flags=re.I)

    # separators → space
    n = n.replace("_", " ").replace("-", " ").replace(".", " ").strip()
    # title case, but keep ALLCAPS tokens (e.g., MLF)
    tokens = [t if t.isupper() else t.capitalize() for t in n.split()]
    n = " ".join(tokens)
    return f"{lr} {n}".strip() if lr else n

def _ellipsis(s: str, n: int | None) -> str:
    if n is None or len(s) <= n:
        return s
    return s[: max(0, n - 1)] + "…"

def _tab20_palette() -> list[str]:
    """Return 20 hex colors from matplotlib's tab20 (or a stable fallback)."""
    try:
        import matplotlib as mpl
        import matplotlib.colors as mcolors

        # Preferred in Matplotlib ≥3.7
        if hasattr(mpl, "colormaps"):
            cmap = mpl.colormaps.get_cmap("tab20")
        else:  # Older Matplotlib
            from matplotlib import cm
            cmap = cm.get_cmap("tab20")

        return [mcolors.to_hex(cmap(i / 20.0)) for i in range(20)]
    except Exception:
        # Stable fallback if matplotlib is unavailable
        return [
            "#1f77b4","#ff7f0e","#2ca02c","#d62728","#9467bd",
            "#8c564b","#e377c2","#7f7f7f","#bcbd22","#17becf",
            "#aec7e8","#ffbb78","#98df8a","#ff9896","#c5b0d5",
            "#c49c94","#f7b6d2","#c7c7c7","#dbdb8d","#9edae5"
        ]

def _assign_colors_if_missing(
    roi_meshes: list[dict] | None,
    fiber_bundles: dict[str, np.ndarray] | None,
    user_fiber_colors: dict[str, str] | None,
) -> tuple[list[dict] | None, dict[str, str]]:
    """
    Ensure ROI meshes have a 'color' and fibers have a color mapping.
    Honors user-specified colors, then cycles tab20 for the rest.
    """
    palette = _tab20_palette()
    color_map: dict[str, str] = {}
    out_fiber_colors: dict[str, str] = {}

    # harvest existing mesh colors (by name)
    if roi_meshes:
        for r in roi_meshes:
            nm = r.get("name", "ROI")
            c = r.get("color")
            if isinstance(c, str):
                color_map.setdefault(nm, c)

    # harvest user fiber colors
    if user_fiber_colors:
        for k, v in user_fiber_colors.items():
            color_map.setdefault(k, v)

    # function to get next palette color
    next_idx = 0
    def _next_color():
        nonlocal next_idx
        c = palette[next_idx % len(palette)]
        next_idx += 1
        return c

    # assign missing mesh colors
    if roi_meshes:
        for r in roi_meshes:
            nm = r.get("name", "ROI")
            if not r.get("color"):
                c = color_map.get(nm) or _next_color()
                r["color"] = c
                color_map[nm] = c

    # assign fiber colors
    if fiber_bundles:
        for nm in fiber_bundles.keys():
            if user_fiber_colors and nm in user_fiber_colors:
                out_fiber_colors[nm] = user_fiber_colors[nm]
            else:
                c = color_map.get(nm) or _next_color()
                color_map[nm] = c
                out_fiber_colors[nm] = c

    return roi_meshes, out_fiber_colors

def _swatch_shape(fig, color, x, y, size=0.018, border="white", valign="middle"):
    # valign: "middle" | "top" | "bottom"
    if valign == "middle":
        y0, y1 = y - size/2, y + size/2
    elif valign == "top":
        y0, y1 = y - size, y
    else:  # "bottom"
        y0, y1 = y, y + size

    fig.add_shape(dict(
        type="rect",
        xref="paper", yref="paper",
        x0=x, y0=y0, x1=x + size, y1=y1,
        line=dict(color=border, width=1),
        fillcolor=color,
        layer="above",
    ))

def _add_camera_snap_controls(
    fig,
    *,
    x: float = 0.02,
    y: float = 0.98,
    font_family: str = "Cambria",
    font_size: int = 12,
):
    """
    Adds small camera snap buttons: R, L, A, P, S, I.
    """
    eyes = {
        "R": dict(x= 2.0, y= 0.0, z= 0.0),  # right lateral
        "L": dict(x=-2.0, y= 0.0, z= 0.0),  # left lateral
        "A": dict(x= 0.0, y= 2.0, z= 0.0),  # anterior (front)
        "P": dict(x= 0.0, y=-2.0, z= 0.0),  # posterior (back)
        "S": dict(x= 0.0, y= 0.0, z= 2.0),  # superior (top)
        "I": dict(x= 0.0, y= 0.0, z=-2.0),  # inferior (bottom)
    }
    buttons = []
    for lbl, eye in eyes.items():
        buttons.append(dict(
            label=lbl,
            method="relayout",
            args=[{"scene.camera": {"eye": eye}}],
        ))
    updatemenus = list(fig.layout.updatemenus) if fig.layout.updatemenus else []
    updatemenus.append(dict(
        type="buttons",
        x=x, y=y,
        xanchor="left", yanchor="top",
        direction="right",
        showactive=False,
        buttons=buttons,
        bgcolor="rgba(245,245,245,0.95)",
        bordercolor="rgba(200,200,200,1)",
        borderwidth=1,
        pad={"r": 4, "l": 4, "t": 2, "b": 2},
        font={"family": font_family, "size": font_size},
    ))
    fig.update_layout(updatemenus=updatemenus)

def add_custom_legend_panel_checkboxes(
    fig,
    group_indices,
    item_indices,
    *,
    subitem_indices=None,
    # anchor of the panel in paper coords
    panel_x: float = 1.02,
    panel_y: float = 1.00,
    # vertical spacing
    vspace_in_group: float = 0.025,       # distance between items/rows inside a group
    vspace_between_groups: float = 0.05,  # extra space after finishing each group
    # horizontal layout
    indent_lvl1: float = 0.012,           # indent for items under a group (2-level)
    indent_lvl2: float = 0.024,           # indent for items under a subgroup (3-level)
    items_per_row: int = 10,              # wrap items horizontally within a subgroup
    hspace_item_col: float = 0.05,        # distance between item columns
    hspace_check_to_swatch: float = 0.010,
    hspace_swatch_to_text: float = 0.035,
    swatch_size: float = 0.016,
    # text & style
    max_item_label_chars: int | None = 18,
    font_family: str = "Cambria",
    font_size: int = 13,
    colorize_item_label: bool = False,
    # ensure there is enough room on the right side (pixels)
    panel_margin_px: int = 280,
):
    """
    Build a checkbox panel with optional nested (group -> subgroup -> items) structure.

    group_indices: dict[str, list[int]]
        Top-level groups -> trace indices.
    item_indices: dict[str, list[tuple[item_label, [trace_idxs...]]]]
        2-level groups -> flat items.
    subitem_indices: dict[str, dict[str, list[tuple[item_label, [trace_idxs...]]]]]
        3-level groups -> { subgroup_label: [(item_label, [trace_idxs...]), ...] }.
    """
    import numpy as np

    subitem_indices = subitem_indices or {}

    # ---------- helpers ----------
    def _infer_item_color(idx_list):
        if not idx_list:
            return "#888"
        tr = fig.data[idx_list[0]]
        col = None
        if getattr(tr, "marker", None) is not None:
            col = getattr(tr.marker, "color", None)
            if isinstance(col, (list, tuple, np.ndarray)):
                col = None
        if not col and getattr(tr, "line", None) is not None:
            col = getattr(tr.line, "color", None)
        if not col and hasattr(tr, "color"):
            col = getattr(tr, "color", None)
        return col or "#888"

    def _checkbox_anno(x, y, checked=True):
        return dict(
            x=x, y=y, xref="paper", yref="paper",
            xanchor="left", yanchor="middle",  
            text="☑" if checked else "☐",
            showarrow=False,
            font=dict(family=font_family, size=font_size+3, color="#1d3557"),
            align="left",
            bgcolor="rgba(248,248,248,0.9)",
            bordercolor="rgba(200,200,200,1)",
            borderwidth=1,
            borderpad=3,
        )

    def _label_anno(text, x, y, bold=False, color="#111"):
        return dict(
            x=x, y=y, xref="paper", yref="paper",
            xanchor="left", yanchor="middle",  
            text=f"<b>{text}</b>" if bold else text,
            showarrow=False,
            font=dict(family=font_family, size=font_size + (1 if bold else 0), color=color),
            align="left",
            bgcolor="rgba(255,255,255,0.0)",
            bordercolor="rgba(0,0,0,0)",
        )

    def _update_boxes_layout(ann_idxs, checked: bool):
        symbol = "☑" if checked else "☐"
        return {f"annotations[{i}].text": symbol for i in ann_idxs}

    # ---------- start ----------
    updatemenus = list(fig.layout.updatemenus) if fig.layout.updatemenus else []
    annotations = list(fig.layout.annotations) if fig.layout.annotations else []

    # base x positions (checkbox → swatch → text)
    x_check = panel_x
    x_swatch_base = x_check + hspace_check_to_swatch
    x_text_base = x_swatch_base + swatch_size + hspace_swatch_to_text

    # current y cursor
    y = panel_y

    group_box_anno_idx: dict[str, int] = {}
    subgroup_box_anno_idx: dict[tuple[str, str], int] = {}
    item_box_anno_idx: dict[tuple[str, str | None, str], int] = {}

    for group in group_indices.keys():
        g_idxs = group_indices.get(group, [])

        # group row
        g_box_idx = len(annotations)
        annotations.append(_checkbox_anno(x_check, y, checked=True))
        annotations.append(_label_anno(group, x_text_base, y, bold=True))
        group_box_anno_idx[group] = g_box_idx

        # placeholder group toggle button
        updatemenus.append(dict(
            type="buttons",
            x=x_check, y=y,
            xanchor="left", yanchor="top",
            direction="right",
            showactive=False,
            buttons=[dict(method="restyle", label=" ", args=[{'visible': True}, g_idxs])],
            bgcolor="rgba(255,255,255,0.0)",
            bordercolor="rgba(0,0,0,0)",
            pad={"r": 0, "l": 0, "t": 0, "b": 0},
        ))
        group_btn_pos = len(updatemenus) - 1
        y -= vspace_in_group

        child_box_idxs_for_group: list[int] = []

        # -------- 3-level: group -> subgroup -> items --------
        if group in subitem_indices:
            for sublabel, sub_items in subitem_indices[group].items():
                # subgroup header
                sg_box_idx = len(annotations)
                annotations.append(_checkbox_anno(x_check + indent_lvl1, y, checked=True))
                annotations.append(_label_anno(sublabel, x_text_base + indent_lvl1, y, bold=False))
                subgroup_box_anno_idx[(group, sublabel)] = sg_box_idx
                child_box_idxs_for_group.append(sg_box_idx)

                # placeholder subgroup button
                updatemenus.append(dict(
                    type="buttons",
                    x=x_check + indent_lvl1, y=y,
                    xanchor="left", yanchor="top",
                    direction="right",
                    showactive=False,
                    buttons=[dict(method="restyle", label=" ", args=[{'visible': True}, []])],
                    bgcolor="rgba(255,255,255,0.0)",
                    bordercolor="rgba(0,0,0,0)",
                    pad={"r": 0, "l": 0, "t": 0, "b": 0},
                ))
                sg_btn_pos = len(updatemenus) - 1

                y -= vspace_in_group  # drop to first item row

                # layout for items under this subgroup
                cols = max(1, int(items_per_row))
                col = 0
                row_y = y
                sg_trace_idxs: list[int] = []
                sg_child_item_box_idxs: list[int] = []

                for (item_label, idxs) in sub_items:
                    x_offset = col * hspace_item_col

                    # checkbox
                    i_box_idx = len(annotations)
                    annotations.append(_checkbox_anno(x_check + indent_lvl2 + x_offset, row_y, checked=True))
                    item_box_anno_idx[(group, sublabel, item_label)] = i_box_idx

                    # swatch + text
                    color = _infer_item_color(idxs)
                    _swatch_shape(fig, color, x_swatch_base + indent_lvl2 + x_offset, row_y, size=swatch_size)
                    label_color = color if colorize_item_label else "#111"
                    annotations.append(_label_anno(
                        _ellipsis(item_label, max_item_label_chars),
                        x_text_base + indent_lvl2 + x_offset, row_y,
                        color=label_color
                    ))

                    # per-item toggle button
                    show_layout = _update_boxes_layout([i_box_idx], True)
                    hide_layout = _update_boxes_layout([i_box_idx], False)
                    updatemenus.append(dict(
                        type="buttons",
                        x=x_check + indent_lvl2 + x_offset, y=row_y,
                        xanchor="left", yanchor="top",
                        direction="right",
                        showactive=False,
                        buttons=[dict(
                            method="update", label=" ",
                            args=[{'visible': [True] * len(idxs)}, show_layout, idxs],
                            args2=[{'visible': [False] * len(idxs)}, hide_layout, idxs],
                        )],
                        bgcolor="rgba(255,255,255,0.0)",
                        bordercolor="rgba(0,0,0,0)",
                        pad={"r": 0, "l": 0, "t": 0, "b": 0},
                    ))

                    sg_trace_idxs.extend(idxs)
                    sg_child_item_box_idxs.append(i_box_idx)
                    child_box_idxs_for_group.append(i_box_idx)

                    # wrap columns
                    col += 1
                    if col >= cols:
                        col = 0
                        row_y -= vspace_in_group

                # advance y to line after the last used row
                y = row_y if col == 0 else (row_y - vspace_in_group)

                # patch subgroup button to toggle its children
                show_layout = _update_boxes_layout([sg_box_idx] + sg_child_item_box_idxs, True)
                hide_layout = _update_boxes_layout([sg_box_idx] + sg_child_item_box_idxs, False)
                updatemenus[sg_btn_pos]["buttons"] = [dict(
                    method="update", label=" ",
                    args=[{'visible': [True] * len(sg_trace_idxs)}, show_layout, sg_trace_idxs],
                    args2=[{'visible': [False] * len(sg_trace_idxs)}, hide_layout, sg_trace_idxs],
                )]

            # add group gap
            y -= vspace_between_groups

        # -------- 2-level: group -> items --------
        else:
            for (item_label, idxs) in item_indices.get(group, []):
                # checkbox
                i_box_idx = len(annotations)
                annotations.append(_checkbox_anno(x_check + indent_lvl1, y, checked=True))
                item_box_anno_idx[(group, None, item_label)] = i_box_idx

                # swatch + text
                color = _infer_item_color(idxs)
                _swatch_shape(fig, color, x_swatch_base + indent_lvl1, y, size=swatch_size)
                label_color = color if colorize_item_label else "#111"
                annotations.append(_label_anno(
                    _ellipsis(item_label, max_item_label_chars),
                    x_text_base + indent_lvl1, y,
                    color=label_color
                ))

                # per-item toggle
                show_layout = _update_boxes_layout([i_box_idx], True)
                hide_layout = _update_boxes_layout([i_box_idx], False)
                updatemenus.append(dict(
                    type="buttons",
                    x=x_check + indent_lvl1, y=y,
                    xanchor="left", yanchor="top",
                    direction="right",
                    showactive=False,
                    buttons=[dict(
                        method="update", label=" ",
                        args=[{'visible': [True] * len(idxs)}, show_layout, idxs],
                        args2=[{'visible': [False] * len(idxs)}, hide_layout, idxs],
                    )],
                    bgcolor="rgba(255,255,255,0.0)",
                    bordercolor="rgba(0,0,0,0)",
                    pad={"r": 0, "l": 0, "t": 0, "b": 0},
                ))
                child_box_idxs_for_group.append(i_box_idx)
                y -= vspace_in_group

            # add group gap
            y -= vspace_between_groups

        # patch group button to toggle all children (subgroups & items)
        all_annos_for_group = [g_box_idx] + child_box_idxs_for_group
        show_layout = _update_boxes_layout(all_annos_for_group, True)
        hide_layout = _update_boxes_layout(all_annos_for_group, False)
        updatemenus[group_btn_pos]["buttons"] = [
            dict(method="update",
                 label=" ",
                 args=[{'visible': [True] * len(g_idxs)}, show_layout, g_idxs],
                 args2=[{'visible': [False] * len(g_idxs)}, hide_layout, g_idxs]),
        ]

    # -------- Global controls --------
    all_trace_idxs = sorted({i for vals in group_indices.values() for i in vals})
    all_box_idxs = (
        list(group_box_anno_idx.values())
        + list(subgroup_box_anno_idx.values())
        + list(item_box_anno_idx.values())
    )

    annotations.append(_label_anno("Global", x_text_base, y, bold=True))
    updatemenus.append(dict(
        type="buttons",
        x=x_text_base, y=y - vspace_in_group,
        xanchor="left", yanchor="top",
        direction="right",
        showactive=False,
        buttons=[
            dict(method="update",
                 label="Show all",
                 args=[{'visible': True}, _update_boxes_layout(all_box_idxs, True), all_trace_idxs]),
            dict(method="update",
                 label="Clear all",
                 args=[{'visible': False}, _update_boxes_layout(all_box_idxs, False), all_trace_idxs]),
        ],
        bgcolor="rgba(245,245,245,0.95)",
        bordercolor="rgba(200,200,200,1)",
        borderwidth=1,
        pad={"r": 6, "l": 6, "t": 2, "b": 2},
        font={"family": font_family, "size": max(11, font_size - 1)},
    ))

    current_r = int(getattr(fig.layout.margin, "r", 0) or 0)
    fig.update_layout(
        margin=dict(r=max(current_r, int(panel_margin_px))),
        updatemenus=updatemenus,
        annotations=annotations,
    )

# ---------- main figure ----------

def build_ieeg_figure(
    *,
    subjects: list[dict] | None = None,
    # --- pials ---
    lh_pial: str | None = None,
    rh_pial: str | None = None,
    show_pials: bool = True,
    # --- MRI slices ---
    t1_path: str | None = None,
    show_slices: bool = True,
    slice_x: int | None = None,
    slice_y: int | None = None,
    slice_z: int | None = None,
    slice_opacity: float = 0.6,
    slice_clim: tuple[float, float] | None = None,
    # --- electrode coloring ---
    color_mode: str = "by_subject",
    subject_color_map: dict[str, str] | None = None,
    electrode_colors: dict | None = None,
    scores_by_subject: dict[str, np.ndarray] | None = None,
    colorscale=None, vmin=None, vmax=None, constant_color="crimson",
    # --- aesthetics ---
    marker_size: int = 9, line_width: int = 9, extension_length: float = 30.0,
    width: int = 1800, height: int = 1400, bgcolor: str = "white",
    camera_eye: dict | None = None, font_family: str = "Cambria", font_size: int = 16,
    # --- ROI meshes ---
    roi_meshes: list[dict] | None = None,
    roi_smooth_iters: int | None = None,
    show_roi_meshes: bool = True,
    fiber_bundles: dict[str, np.ndarray] | None = None,
    fiber_colors: dict[str, str] | None = None,
    fiber_line_width: int = 8,
    fiber_opacity: float | None = None, 
    # --- FreeSurfer aseg support ---
    freesurfer_aseg: str | None = None,
    fs_roi_labels: dict[str, int] | None = None,
    fs_roi_colors: dict[str, str] | None = None,
    fs_roi_opacity: float = 0.25,
    fs_roi_smoothing: bool = True,
    # --- Legend/panel layout ---
    panel_x: float = 1.015,
    panel_y: float = 1.00,
    vspace_in_group: float = 0.03,
    vspace_between_groups: float = 0.02,
    indent_lvl1: float = 0.012,
    indent_lvl2: float = 0.024,
    items_per_row: int = 5,
    hspace_item_col: float = 0.06,
    hspace_check_to_swatch: float = 0.022,
    hspace_swatch_to_text: float = 0.0022,
    swatch_size: float = 0.016,
    max_item_label_chars: int | None = 18,
    panel_margin_px: int = 400,
    colorize_item_label: bool = True,
    # --- camera snap buttons ---
    add_view_buttons: bool = True,
    view_buttons_x: float = 0.02,
    view_buttons_y: float = 0.98,
):

    go = _require_plotly()
    from collections import defaultdict, OrderedDict

    if colorscale is None:
        colorscale = [[0.0, "green"], [0.5, "orange"], [1.0, "red"]]

    group_indices: dict[str, list[int]] = defaultdict(list)            # group -> [trace_idx...]
    item_indices: dict[str, list[tuple[str, list[int]]]] = defaultdict(list)  # group -> [(label, [trace_idx...]), ...]

    # --- figure & scene ---
    fig = go.Figure()
    scene = dict(
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        zaxis=dict(visible=False),
        aspectmode="data",
        bgcolor=bgcolor,
        camera=(camera_eye or dict(eye=dict(x=-1.5, y=0, z=0))),
    )
    fig.update_layout(
        width=width,
        height=height,
        scene=scene,
        showlegend=False,
        font=dict(family=font_family, size=font_size),
    )

    # =====================================================================
    # PIALS (3D model)
    # =====================================================================
    if show_pials and (lh_pial or rh_pial):
        l_mesh, r_mesh = load_pial_mesh(lh_pial, rh_pial)
        group = "3D model"
        if l_mesh is not None:
            lv, lf = l_mesh
            fig.add_trace(go.Mesh3d(
                x=lv[:, 0], y=lv[:, 1], z=lv[:, 2],
                i=lf[:, 0], j=lf[:, 1], k=lf[:, 2],
                color="lightgray", opacity=0.12, name="Left hemisphere",
                lighting=dict(ambient=0.2, diffuse=0.1, specular=0.5, roughness=0.0),
            ))
            idx = len(fig.data) - 1
            group_indices[group].append(idx)
            item_indices[group].append(("Left hemisphere", [idx]))
        if r_mesh is not None:
            rv, rf = r_mesh
            fig.add_trace(go.Mesh3d(
                x=rv[:, 0], y=rv[:, 1], z=rv[:, 2],
                i=rf[:, 0], j=rf[:, 1], k=rf[:, 2],
                color="lightgray", opacity=0.12, name="Right hemisphere",
                lighting=dict(ambient=0.2, diffuse=0.1, specular=0.5, roughness=0.0),
            ))
            idx = len(fig.data) - 1
            group_indices[group].append(idx)
            item_indices[group].append(("Right hemisphere", [idx]))

    # =====================================================================
    # MRI SLICES
    # =====================================================================
    if show_slices and t1_path:
        img = nb.load(t1_path)
        sx, sy, sz = img.shape
        if slice_x is None: slice_x = sx // 2
        if slice_y is None: slice_y = sy // 2
        if slice_z is None: slice_z = sz // 2

        sagittal, axial, coronal, reference_data = extract_mni_slices(
            reference_file=t1_path, slice_x=slice_x, slice_y=slice_y, slice_z=slice_z
        )

        group = "MRI slices"

        if axial is not None:
            X_ax, Y_ax, Z_ax = axial
            fig.add_trace(go.Surface(
                z=Z_ax, x=X_ax, y=Y_ax,
                surfacecolor=reference_data[:, slice_y, :].T,
                colorscale="gray", opacity=slice_opacity,
                cmin=(slice_clim[0] if slice_clim else None),
                cmax=(slice_clim[1] if slice_clim else None),
                showscale=False, name="Axial",
            ))
            idx = len(fig.data) - 1
            group_indices[group].append(idx)
            item_indices[group].append(("Axial", [idx]))

        if coronal is not None:
            X_cor, Y_cor, Z_cor = coronal
            fig.add_trace(go.Surface(
                y=Y_cor, x=X_cor, z=Z_cor,
                surfacecolor=reference_data[:, :, slice_z].T,
                colorscale="gray", opacity=slice_opacity,
                cmin=(slice_clim[0] if slice_clim else None),
                cmax=(slice_clim[1] if slice_clim else None),
                showscale=False, name="Coronal",
            ))
            idx = len(fig.data) - 1
            group_indices[group].append(idx)
            item_indices[group].append(("Coronal", [idx]))

        if sagittal is not None:
            X_sag, Y_sag, Z_sag = sagittal
            fig.add_trace(go.Surface(
                x=X_sag, y=Y_sag, z=Z_sag,
                surfacecolor=reference_data[slice_x, :, :].T,
                colorscale="gray", opacity=slice_opacity,
                cmin=(slice_clim[0] if slice_clim else None),
                cmax=(slice_clim[1] if slice_clim else None),
                showscale=False, name="Sagittal",
            ))
            idx = len(fig.data) - 1
            group_indices[group].append(idx)
            item_indices[group].append(("Sagittal", [idx]))

    # =====================================================================
    # ELECTRODES (multi-subject) -- subjects optional
    # =====================================================================
    subitem_indices: dict[str, dict[str, list[tuple[str, list[int]]]]] = {}
    if subjects:
        bundle = read_subject_electrodes(subjects)
        subj_data = {}
        for subj, payload in bundle.items():
            C = payload["coords"]
            N = payload["names"]
            subj_data[subj] = {
                "coords": C,
                "names": N,
                "color": (subject_color_map or {}).get(subj, payload.get("color", "dodgerblue")),
            }

        # normalize cmap range
        if color_mode == "cmap" and scores_by_subject:
            flat = np.concatenate([np.asarray(v, float) for v in scores_by_subject.values()]) \
                   if scores_by_subject else np.array([])
            finite = flat[np.isfinite(flat)]
            if finite.size:
                if vmin is None: vmin = float(np.nanmin(finite))
                if vmax is None: vmax = float(np.nanmax(finite))
                if vmin == vmax:
                    vmin, vmax = vmin - 1.0, vmax + 1.0

        for subj, payload in subj_data.items():
            C, N = payload["coords"], payload["names"]
            subj_color = payload["color"]
            group = f"Electrodes — subject {subj}"

            names = np.array([f"{subj}:{n}" for n in N], dtype=object)
            by_el = OrderedDict()
            for i, nm in enumerate(names):
                by_el.setdefault(nm, []).append(i)

            subitem_indices[group] = {"Contacts": [], "Trajectories": []}

            cbar_added = False
            for gname, idxs in by_el.items():
                P = C[idxs]
                marker = dict(size=marker_size, opacity=1.0, line=dict(width=0.8, color="black"))
                if color_mode == "cmap":
                    s_subj = np.asarray(scores_by_subject.get(subj, np.full(C.shape[0], np.nan)), float)
                    s = s_subj[idxs]
                    marker.update(dict(color=s, colorscale=colorscale, cmin=vmin, cmax=vmax))
                    if not cbar_added:
                        marker["colorbar"] = dict(title=dict(text="#HFOs"))
                        cbar_added = True
                elif color_mode == "by_subject":
                    marker.update(dict(color=subj_color))
                elif color_mode == "by_electrode":
                    marker.update(dict(color=(electrode_colors or {}).get(gname, "dodgerblue")))
                elif color_mode == "constant":
                    marker.update(dict(color=constant_color))
                else:
                    raise ValueError("color_mode must be one of {'by_subject','by_electrode','constant','cmap'}")

                pretty = _pretty_elname(gname)

                # Contacts trace
                fig.add_trace(go.Scatter3d(
                    x=P[:, 0], y=P[:, 1], z=P[:, 2],
                    mode="markers",
                    name=f"{subj}: {pretty} contacts",
                    marker=marker,
                ))
                c_idx = len(fig.data) - 1
                group_indices[group].append(c_idx)
                subitem_indices[group]["Contacts"].append((f"{pretty}", [c_idx]))

                # Trajectory trace (if >=2 contacts)
                if P.shape[0] >= 2:
                    v = P[-1] - P[0]
                    n = np.linalg.norm(v)
                    if n > 0:
                        v = v / n
                        deep = P[-1] + extension_length * v
                        TP = np.vstack([P, deep])
                        if color_mode == "by_electrode":
                            ln_color = (electrode_colors or {}).get(gname, "dodgerblue")
                        elif color_mode == "constant":
                            ln_color = constant_color
                        elif color_mode == "by_subject":
                            ln_color = subj_color
                        else:
                            ln_color = "dodgerblue"
                        fig.add_trace(go.Scatter3d(
                            x=TP[:, 0], y=TP[:, 1], z=TP[:, 2],
                            mode="lines",
                            name=f"{subj}: {pretty} trajectory",
                            line=dict(width=line_width, color=ln_color),
                            hoverinfo="skip",
                        ))
                        t_idx = len(fig.data) - 1
                        group_indices[group].append(t_idx)
                        subitem_indices[group]["Trajectories"].append((f"{pretty}", [t_idx]))

    # =====================================================================
    # ROI meshes + fiber bundles  (deterministic colors + fiber opacity)
    # =====================================================================
    if show_roi_meshes:
        merged_roi_meshes: list[dict] = list(roi_meshes or [])

        # FS aseg -> optional merge
        if freesurfer_aseg and fs_roi_labels:
            merged_roi_meshes.extend(
                _fs_aseg_to_mesh_list(
                    freesurfer_aseg,
                    fs_roi_labels,
                    colors=fs_roi_colors,
                    opacity=fs_roi_opacity,
                    smoothing=fs_roi_smoothing,
                )
            )

        # Assign colors (only where missing) using tab20; fibers get colors too
        merged_roi_meshes, fiber_colors_final = _assign_colors_if_missing(
            merged_roi_meshes or [],
            fiber_bundles,
            fiber_colors,
        )

        if merged_roi_meshes or fiber_bundles:
            group = "ROI meshes"

            def _maybe_smooth(v, f, iters):
                try:
                    return smooth_trimesh(v, f, iterations=iters) if (iters and iters > 0) else (v, f)
                except NameError:
                    return (v, f)

            # ---- Mesh ROIs ----
            for r in merged_roi_meshes:
                v = np.asarray(r["vertices"], float)
                f = np.asarray(r["faces"], int)
                v, f = _maybe_smooth(v, f, roi_smooth_iters)

                display_name = pretty_roi_name(r.get("name", "ROI"))
                face_color = r.get("color", "tomato")
                face_opacity = float(r.get("opacity", fs_roi_opacity))

                fig.add_trace(go.Mesh3d(
                    x=v[:, 0], y=v[:, 1], z=v[:, 2],
                    i=f[:, 0], j=f[:, 1], k=f[:, 2],
                    color=face_color,
                    opacity=face_opacity,
                    name=display_name,
                    lighting=dict(ambient=0.25, diffuse=0.3, specular=0.2),
                ))
                idx = len(fig.data) - 1
                group_indices[group].append(idx)
                item_indices[group].append((display_name, [idx]))

            # ---- Fibers (.mat) ----
            if fiber_bundles:
                # default fiber opacity: follow mesh default if not specified
                f_opac = fs_roi_opacity if fiber_opacity is None else float(fiber_opacity)
                for raw_name, arr in fiber_bundles.items():
                    M = np.asarray(arr)
                    if M.ndim != 2 or M.shape[1] < 3:
                        continue

                    pts = M[:, :3]
                    ids = M[:, 3] if M.shape[1] >= 4 else np.zeros(len(pts))

                    # build a single polyline with None breaks between fiber_ids
                    order = np.argsort(ids, kind="mergesort")
                    pts = pts[order]; ids = ids[order]
                    breaks = np.where(np.diff(ids) != 0)[0] + 1

                    x = []; y = []; z = []
                    last = 0
                    for b in np.r_[breaks, len(pts)]:
                        seg = pts[last:b]
                        if seg.size:
                            x += seg[:, 0].tolist()
                            y += seg[:, 1].tolist()
                            z += seg[:, 2].tolist()
                            x.append(None); y.append(None); z.append(None)
                        last = b

                    col = (fiber_colors_final or {}).get(raw_name, "#bbbbbb")
                    display_name = pretty_roi_name(raw_name)
                    fig.add_trace(go.Scatter3d(
                        x=x, y=y, z=z,
                        mode="lines",
                        line=dict(width=fiber_line_width, color=col),
                        name=display_name,
                        hoverinfo="skip",
                        opacity=f_opac,
                    ))
                    idx = len(fig.data) - 1
                    group_indices[group].append(idx)
                    item_indices[group].append((display_name, [idx]))

    # =====================================================================
    # Custom legend-like panel on the right
    # =====================================================================
    add_custom_legend_panel_checkboxes(
        fig, group_indices, item_indices,
        subitem_indices=subitem_indices,
        panel_x=panel_x, panel_y=panel_y,
        vspace_in_group=vspace_in_group,
        vspace_between_groups=vspace_between_groups,
        indent_lvl1=indent_lvl1,
        indent_lvl2=indent_lvl2,
        items_per_row=items_per_row,
        hspace_item_col=hspace_item_col,
        hspace_check_to_swatch=hspace_check_to_swatch,
        hspace_swatch_to_text=hspace_swatch_to_text,
        swatch_size=swatch_size,
        max_item_label_chars=max_item_label_chars,
        font_family=font_family,
        font_size=font_size,
        colorize_item_label=colorize_item_label,
        panel_margin_px=panel_margin_px,
    )

    # camera snap controls
    if add_view_buttons:
        _add_camera_snap_controls(
            fig,
            x=view_buttons_x, y=view_buttons_y,
            font_family=font_family, font_size=max(11, font_size - 2),
        )

    return fig