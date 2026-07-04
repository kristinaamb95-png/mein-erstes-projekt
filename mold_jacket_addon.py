bl_info = {
    "name": "Mold Jacket Generator (Опалубка для силиконовых форм)",
    "author": "mein-erstes-projekt",
    "version": (1, 0, 0),
    "blender": (3, 3, 0),
    "location": "View3D > Sidebar > Mold Jacket",
    "description": (
        "Generates a splittable rigid jacket (formwork / опалубка) around a master "
        "model for pouring a silicone mold: an offset cavity gap, wall thickness, "
        "a stepped base plate, and split parts with wings and mama/papa alignment locks."
    ),
    "category": "Mesh",
}

import bpy
import bmesh
import math
from mathutils import Vector
from bpy.props import (
    FloatProperty,
    IntProperty,
    EnumProperty,
    PointerProperty,
    BoolProperty,
)
from bpy.types import PropertyGroup, Operator, Panel


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------

def get_or_create_collection(name):
    coll = bpy.data.collections.get(name)
    if coll is None:
        coll = bpy.data.collections.new(name)
        bpy.context.scene.collection.children.link(coll)
    return coll


def link_only(obj, collection):
    for c in list(obj.users_collection):
        c.objects.unlink(obj)
    collection.objects.link(obj)


def duplicate_mesh_only(obj, name, collection):
    """Duplicate obj's mesh (world-space baked) into a new object, no modifiers."""
    depsgraph = bpy.context.evaluated_depsgraph_get()
    eval_obj = obj.evaluated_get(depsgraph)
    new_mesh = bpy.data.meshes.new_from_object(eval_obj, depsgraph=depsgraph)
    new_mesh.name = name
    new_mesh.transform(obj.matrix_world)
    new_obj = bpy.data.objects.new(name, new_mesh)
    collection.objects.link(new_obj)
    return new_obj


def apply_modifiers(obj):
    """Bake all modifiers on obj into its mesh data (context/UI independent)."""
    depsgraph = bpy.context.evaluated_depsgraph_get()
    eval_obj = obj.evaluated_get(depsgraph)
    new_mesh = bpy.data.meshes.new_from_object(eval_obj, depsgraph=depsgraph)
    old_mesh = obj.data
    obj.modifiers.clear()
    obj.data = new_mesh
    if old_mesh.users == 0:
        bpy.data.meshes.remove(old_mesh)
    return obj


def keep_only_largest_island(obj):
    """If obj's mesh has multiple disconnected pieces, keep only the one with the
    largest bounding-box volume and delete the rest (used after an outward
    Solidify pass, which leaves the original-position surface as a separate,
    smaller-radius island)."""
    bm = bmesh.new()
    bm.from_mesh(obj.data)
    bm.faces.ensure_lookup_table()

    visited = set()
    islands = []
    for seed in bm.faces:
        if seed.index in visited:
            continue
        stack = [seed]
        visited.add(seed.index)
        island = []
        while stack:
            f = stack.pop()
            island.append(f)
            for e in f.edges:
                for lf in e.link_faces:
                    if lf.index not in visited:
                        visited.add(lf.index)
                        stack.append(lf)
        islands.append(island)

    if len(islands) > 1:
        def bbox_volume(island):
            xs, ys, zs = [], [], []
            for f in island:
                for v in f.verts:
                    xs.append(v.co.x)
                    ys.append(v.co.y)
                    zs.append(v.co.z)
            return (max(xs) - min(xs)) * (max(ys) - min(ys)) * (max(zs) - min(zs))

        best = max(islands, key=bbox_volume)
        keep = set(f.index for f in best)
        to_delete = [f for f in bm.faces if f.index not in keep]
        bmesh.ops.delete(bm, geom=to_delete, context='FACES_ONLY')
        # also drop now-isolated verts/edges left over from deleted faces
        loose_verts = [v for v in bm.verts if not v.link_faces]
        bmesh.ops.delete(bm, geom=loose_verts, context='VERTS')

    bm.to_mesh(obj.data)
    bm.free()
    obj.data.update()


def recalc_normals(obj):
    bm = bmesh.new()
    bm.from_mesh(obj.data)
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
    bm.to_mesh(obj.data)
    bm.free()
    obj.data.update()


def offset_filled_solid(source_obj, distance, voxel_size, name, collection):
    """Return a new watertight solid object: source_obj's boundary surface grown
    outward by `distance` (may be 0), cleaned into a single simple blob via voxel
    remesh. Self-intersections created in concave regions by the outward offset
    are naturally merged away by the voxel remesh pass."""
    dup = duplicate_mesh_only(source_obj, name, collection)

    if distance > 1e-9:
        sm = dup.modifiers.new("Grow", type='SOLIDIFY')
        sm.thickness = distance
        sm.offset = 1.0
        sm.use_even_offset = True
        sm.use_quality_normals = True
        apply_modifiers(dup)
        keep_only_largest_island(dup)

    rm = dup.modifiers.new("Clean", type='REMESH')
    rm.mode = 'VOXEL'
    rm.voxel_size = voxel_size
    rm.use_remove_disconnected = False
    apply_modifiers(dup)
    recalc_normals(dup)
    return dup


def clean_solid(obj, voxel_size):
    """Voxel-remesh cleanup pass: chained EXACT boolean operations can leave a
    handful of degenerate/non-manifold faces behind, which then makes the
    *next* boolean operation on this object fail unpredictably (producing
    garbage geometry). Re-remeshing after each major boolean stage keeps the
    mesh solid and safe to feed into further booleans."""
    rm = obj.modifiers.new("Clean", type='REMESH')
    rm.mode = 'VOXEL'
    rm.voxel_size = voxel_size
    rm.use_remove_disconnected = False
    apply_modifiers(obj)
    recalc_normals(obj)
    return obj


def boolean_combine(obj_a, obj_b, operation, delete_b=True, solver='EXACT'):
    """obj_a = obj_a <operation> obj_b. Returns obj_a. Deletes obj_b if requested."""
    mod = obj_a.modifiers.new("Bool_" + operation.title(), type='BOOLEAN')
    mod.operation = operation
    mod.object = obj_b
    mod.solver = solver
    apply_modifiers(obj_a)
    if delete_b:
        mesh_b = obj_b.data
        bpy.data.objects.remove(obj_b, do_unlink=True)
        if mesh_b.users == 0:
            bpy.data.meshes.remove(mesh_b)
    return obj_a


def remove_object(obj):
    mesh = obj.data
    bpy.data.objects.remove(obj, do_unlink=True)
    if mesh and mesh.users == 0:
        bpy.data.meshes.remove(mesh)


def world_bounds(obj):
    corners = [obj.matrix_world @ Vector(c) for c in obj.bound_box]
    xs = [c.x for c in corners]
    ys = [c.y for c in corners]
    zs = [c.z for c in corners]
    return (min(xs), min(ys), min(zs)), (max(xs), max(ys), max(zs))


def world_vert_coords(obj):
    mw = obj.matrix_world
    return [mw @ v.co for v in obj.data.vertices]


def make_box_from_bounds(name, xmin, xmax, ymin, ymax, zmin, zmax, collection):
    mesh = bpy.data.meshes.new(name)
    obj = bpy.data.objects.new(name, mesh)
    collection.objects.link(obj)
    bm = bmesh.new()
    verts = [
        bm.verts.new((xmin, ymin, zmin)),
        bm.verts.new((xmax, ymin, zmin)),
        bm.verts.new((xmax, ymax, zmin)),
        bm.verts.new((xmin, ymax, zmin)),
        bm.verts.new((xmin, ymin, zmax)),
        bm.verts.new((xmax, ymin, zmax)),
        bm.verts.new((xmax, ymax, zmax)),
        bm.verts.new((xmin, ymax, zmax)),
    ]
    faces = [
        (0, 1, 2, 3), (4, 7, 6, 5),
        (0, 4, 5, 1), (1, 5, 6, 2),
        (2, 6, 7, 3), (3, 7, 4, 0),
    ]
    for f in faces:
        bm.faces.new([verts[i] for i in f])
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
    bm.to_mesh(mesh)
    bm.free()
    return obj


def make_wedge_prism(name, center_xy, angle_start, angle_end, radius, z_min, z_max, collection):
    """A solid pie-slice prism (vertical, along Z) spanning [angle_start, angle_end]
    around center_xy, used as a boolean cutter to split the jacket into parts."""
    span = angle_end - angle_start
    segments = max(2, int(math.ceil(math.degrees(span) / 15.0)))
    angles = [angle_start + span * i / segments for i in range(segments + 1)]

    mesh = bpy.data.meshes.new(name)
    obj = bpy.data.objects.new(name, mesh)
    collection.objects.link(obj)
    bm = bmesh.new()

    cx, cy = center_xy
    rim_bottom = [bm.verts.new((cx + radius * math.cos(a), cy + radius * math.sin(a), z_min)) for a in angles]
    rim_top = [bm.verts.new((cx + radius * math.cos(a), cy + radius * math.sin(a), z_max)) for a in angles]
    center_bottom = bm.verts.new((cx, cy, z_min))
    center_top = bm.verts.new((cx, cy, z_max))

    # bottom fan
    for i in range(segments):
        bm.faces.new((center_bottom, rim_bottom[i], rim_bottom[i + 1]))
    # top fan
    for i in range(segments):
        bm.faces.new((center_top, rim_top[i + 1], rim_top[i]))
    # outer arc wall
    for i in range(segments):
        bm.faces.new((rim_bottom[i], rim_bottom[i + 1], rim_top[i + 1], rim_top[i]))
    # two flat radial side faces (the actual seam cut faces)
    bm.faces.new((center_bottom, rim_bottom[0], rim_top[0], center_top))
    bm.faces.new((center_bottom, center_top, rim_top[segments], rim_bottom[segments]))

    bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
    bm.to_mesh(mesh)
    bm.free()
    return obj


def make_cylinder(name, center, radius, depth, axis, collection, segments=24):
    """A solid cylinder centered at `center`, extending +/- depth/2 along `axis`
    (a unit Vector)."""
    axis = axis.normalized()
    # build an orthonormal basis around axis
    ref = Vector((1, 0, 0)) if abs(axis.z) < 0.9 else Vector((0, 1, 0))
    tangent = axis.cross(ref).normalized()
    bitangent = axis.cross(tangent).normalized()

    mesh = bpy.data.meshes.new(name)
    obj = bpy.data.objects.new(name, mesh)
    collection.objects.link(obj)
    bm = bmesh.new()

    bottom = center - axis * (depth / 2.0)
    top = center + axis * (depth / 2.0)
    ring_bottom = []
    ring_top = []
    for i in range(segments):
        ang = 2 * math.pi * i / segments
        offset = tangent * (radius * math.cos(ang)) + bitangent * (radius * math.sin(ang))
        ring_bottom.append(bm.verts.new(bottom + offset))
        ring_top.append(bm.verts.new(top + offset))
    cb = bm.verts.new(bottom)
    ct = bm.verts.new(top)
    for i in range(segments):
        j = (i + 1) % segments
        bm.faces.new((cb, ring_bottom[i], ring_bottom[j]))
        bm.faces.new((ct, ring_top[j], ring_top[i]))
        bm.faces.new((ring_bottom[i], ring_bottom[j], ring_top[j], ring_top[i]))

    bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
    bm.to_mesh(mesh)
    bm.free()
    return obj


def make_sphere(name, center, radius, collection, segments=16, rings=8):
    mesh = bpy.data.meshes.new(name)
    obj = bpy.data.objects.new(name, mesh)
    collection.objects.link(obj)
    bm = bmesh.new()
    bmesh.ops.create_uvsphere(bm, u_segments=segments, v_segments=rings, radius=radius)
    for v in bm.verts:
        v.co += center
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
    bm.to_mesh(mesh)
    bm.free()
    return obj


def footprint_radius(coords, center_xy, z_min=None, z_max=None):
    cx, cy = center_xy
    best = 0.0
    for co in coords:
        if z_min is not None and co.z < z_min:
            continue
        if z_max is not None and co.z > z_max:
            continue
        r = math.hypot(co.x - cx, co.y - cy)
        if r > best:
            best = r
    return best


def working_master(settings):
    return settings.prepared_object if settings.prepared_object else settings.master_object


def compute_pin_positions(settings, master_obj):
    coords = world_vert_coords(master_obj)
    minb, maxb = world_bounds(master_obj)
    cx = (minb[0] + maxb[0]) / 2.0
    cy = (minb[1] + maxb[1]) / 2.0
    base_z = minb[2]
    model_radius = footprint_radius(coords, (cx, cy), z_min=base_z, z_max=base_z + settings.step1_width)
    pin_radius = model_radius + settings.step2_width * 0.5
    positions = []
    for i in range(settings.pin_count):
        ang = 2 * math.pi * i / max(1, settings.pin_count)
        positions.append((cx + pin_radius * math.cos(ang), cy + pin_radius * math.sin(ang), ang))
    return (cx, cy), base_z, model_radius, positions


# ---------------------------------------------------------------------------
# Property group
# ---------------------------------------------------------------------------

class MoldJacketSettings(PropertyGroup):
    master_object: PointerProperty(
        name="Master Model",
        description="The sculpted master model (мастер-молд) to build the jacket around",
        type=bpy.types.Object,
    )
    prepared_object: PointerProperty(
        name="Prepared Master (internal)",
        type=bpy.types.Object,
    )
    cut_z: FloatProperty(
        name="Cut Height (Z)",
        description="World Z height at which the bottom of the master model is removed (opens the base, no cap)",
        subtype='DISTANCE', unit='LENGTH', default=0.0,
    )

    gap_distance: FloatProperty(
        name="Silicone Gap",
        description="Distance between the master model surface and the inside of the jacket - this is where the silicone is poured (typically ~4mm)",
        subtype='DISTANCE', unit='LENGTH', default=0.004, min=0.0002, max=0.05,
    )
    wall_thickness: FloatProperty(
        name="Jacket Wall Thickness",
        subtype='DISTANCE', unit='LENGTH', default=0.003, min=0.001, max=0.05,
    )
    voxel_size: FloatProperty(
        name="Detail (Voxel Size)",
        description="Resolution used to rebuild the offset surfaces. Smaller = more detail but slower",
        subtype='DISTANCE', unit='LENGTH', default=0.0015, min=0.0003, max=0.01,
    )
    top_margin: FloatProperty(
        name="Top Collar Margin",
        description="Extra height the open jacket top extends above the tallest point of the master model",
        subtype='DISTANCE', unit='LENGTH', default=0.008, min=0.0, max=0.1,
    )

    num_parts: IntProperty(
        name="Number of Parts",
        description="How many pieces the jacket is split into (2 = classic clamshell)",
        default=2, min=2, max=6,
    )
    split_angle_offset: FloatProperty(
        name="Split Angle Offset",
        subtype='ANGLE', default=0.0,
    )

    wings_per_seam: IntProperty(name="Wings per Seam", default=2, min=1, max=6)
    wing_width: FloatProperty(name="Wing Width (outward)", subtype='DISTANCE', unit='LENGTH', default=0.02, min=0.002, max=0.1)
    wing_span: FloatProperty(name="Wing Height", subtype='DISTANCE', unit='LENGTH', default=0.03, min=0.005, max=0.15)
    wing_thickness: FloatProperty(name="Wing Thickness (per side)", subtype='DISTANCE', unit='LENGTH', default=0.004, min=0.001, max=0.02)
    wing_hole_diameter: FloatProperty(
        name="Wing Clamp Hole Diameter",
        description="Diameter of the hole through the wings for a bolt/clip/rubber band. 0 disables it",
        subtype='DISTANCE', unit='LENGTH', default=0.005, min=0.0, max=0.02,
    )

    locks_per_seam: IntProperty(name="Locks per Seam", default=3, min=0, max=10)
    lock_diameter: FloatProperty(name="Lock Ball Diameter", subtype='DISTANCE', unit='LENGTH', default=0.009, min=0.002, max=0.03)
    lock_depth: FloatProperty(name="Lock Protrusion", subtype='DISTANCE', unit='LENGTH', default=0.005, min=0.001, max=0.02)
    lock_clearance: FloatProperty(
        name="Lock Fit Clearance",
        description="Extra radius added to the socket (mama) side so the ball (papa) side fits without binding",
        subtype='DISTANCE', unit='LENGTH', default=0.0003, min=0.0, max=0.002,
    )

    base_margin: FloatProperty(name="Base Plate Margin", subtype='DISTANCE', unit='LENGTH', default=0.008, min=0.0, max=0.1)
    base_plate_thickness: FloatProperty(name="Base Plate Thickness", subtype='DISTANCE', unit='LENGTH', default=0.004, min=0.001, max=0.02)
    step1_height: FloatProperty(name="Inner Step Height", subtype='DISTANCE', unit='LENGTH', default=0.003, min=0.0005, max=0.02)
    step1_width: FloatProperty(name="Inner Step Width", subtype='DISTANCE', unit='LENGTH', default=0.006, min=0.001, max=0.05)
    step2_height: FloatProperty(name="Outer Step Height", subtype='DISTANCE', unit='LENGTH', default=0.003, min=0.0005, max=0.02)
    step2_width: FloatProperty(name="Outer Step Width", subtype='DISTANCE', unit='LENGTH', default=0.004, min=0.001, max=0.05)
    pin_count: IntProperty(name="Alignment Pin Count", default=4, min=0, max=12)
    pin_diameter: FloatProperty(name="Pin Diameter", subtype='DISTANCE', unit='LENGTH', default=0.004, min=0.001, max=0.02)
    pin_height: FloatProperty(name="Pin Height", subtype='DISTANCE', unit='LENGTH', default=0.004, min=0.001, max=0.02)


# ---------------------------------------------------------------------------
# Operator: Prepare Master (cut off the bottom, no cap - открытое дно)
# ---------------------------------------------------------------------------

class MOLDJACKET_OT_prepare_master(Operator):
    bl_idname = "moldjacket.prepare_master"
    bl_label = "Cut Bottom (Prepare Master)"
    bl_description = "Duplicate the master model and remove everything below the Cut Height, leaving an open (uncapped) bottom"
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        s = context.scene.mold_jacket
        return s.master_object is not None and s.master_object.type == 'MESH'

    def execute(self, context):
        settings = context.scene.mold_jacket
        master = settings.master_object
        coll = get_or_create_collection(f"MoldJacket - {master.name}")

        prepared = duplicate_mesh_only(master, f"{master.name}_MoldSource", coll)

        bm = bmesh.new()
        bm.from_mesh(prepared.data)
        geom = list(bm.verts) + list(bm.edges) + list(bm.faces)
        bmesh.ops.bisect_plane(
            bm, geom=geom, dist=1e-6,
            plane_co=(0.0, 0.0, settings.cut_z),
            plane_no=(0.0, 0.0, 1.0),
            clear_inner=True, clear_outer=False,
        )
        bm.to_mesh(prepared.data)
        bm.free()
        prepared.data.update()

        settings.prepared_object = prepared
        self.report({'INFO'}, f"Prepared master created: {prepared.name}")
        return {'FINISHED'}


class MOLDJACKET_OT_set_cut_to_min(Operator):
    bl_idname = "moldjacket.set_cut_to_min"
    bl_label = "Set Cut Height to Model Bottom"
    bl_description = "Set the Cut Height field to the master model's lowest point"

    @classmethod
    def poll(cls, context):
        s = context.scene.mold_jacket
        return s.master_object is not None and s.master_object.type == 'MESH'

    def execute(self, context):
        settings = context.scene.mold_jacket
        minb, _ = world_bounds(settings.master_object)
        settings.cut_z = minb[2]
        return {'FINISHED'}


# ---------------------------------------------------------------------------
# Operator: Generate stepped base plate with alignment pins
# ---------------------------------------------------------------------------

class MOLDJACKET_OT_generate_base_plate(Operator):
    bl_idname = "moldjacket.generate_base_plate"
    bl_label = "Generate Base Plate"
    bl_description = "Build a stepped pedestal plate under the master model footprint with alignment pins for the jacket"
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        s = context.scene.mold_jacket
        return working_master(s) is not None

    def execute(self, context):
        settings = context.scene.mold_jacket
        master = working_master(settings)
        coll = get_or_create_collection(f"MoldJacket - {settings.master_object.name}")

        (cx, cy), base_z, model_radius, pin_positions = compute_pin_positions(settings, master)

        disc_r = model_radius + settings.base_margin
        step2_r = model_radius + settings.step2_width
        step1_r = model_radius + max(0.0005, settings.step1_width * 0.02)

        z_step1_top = base_z
        z_step1_bot = z_step1_top - settings.step1_height
        z_step2_bot = z_step1_bot - settings.step2_height
        z_disc_bot = z_step2_bot - settings.base_plate_thickness

        plate = make_cylinder(
            "BasePlate_Disc", Vector((cx, cy, (z_disc_bot + z_step2_bot) / 2.0)),
            disc_r, z_step2_bot - z_disc_bot, Vector((0, 0, 1)), coll, segments=48,
        )
        step2 = make_cylinder(
            "BasePlate_Step2", Vector((cx, cy, (z_step2_bot + z_step1_bot) / 2.0)),
            step2_r, z_step1_bot - z_step2_bot, Vector((0, 0, 1)), coll, segments=48,
        )
        step1 = make_cylinder(
            "BasePlate_Step1", Vector((cx, cy, (z_step1_bot + z_step1_top) / 2.0)),
            step1_r, z_step1_top - z_step1_bot, Vector((0, 0, 1)), coll, segments=48,
        )

        plate = boolean_combine(plate, step2, 'UNION', delete_b=True)
        plate = boolean_combine(plate, step1, 'UNION', delete_b=True)

        for i, (px, py, _ang) in enumerate(pin_positions):
            pin = make_cylinder(
                f"BasePlate_Pin{i}",
                Vector((px, py, z_step1_bot + settings.pin_height / 2.0)),
                settings.pin_diameter / 2.0, settings.pin_height,
                Vector((0, 0, 1)), coll, segments=16,
            )
            plate = boolean_combine(plate, pin, 'UNION', delete_b=True)

        plate.name = f"{settings.master_object.name}_BasePlate"
        recalc_normals(plate)
        self.report({'INFO'}, f"Base plate created: {plate.name}")
        return {'FINISHED'}


# ---------------------------------------------------------------------------
# Operator: Generate the mold jacket (cavity gap + wall), split into parts
# with wings and mama/papa alignment locks.
# ---------------------------------------------------------------------------

def seam_tangent(angle):
    return Vector((-math.sin(angle), math.cos(angle), 0.0))


def add_wing(part_obj, coll, center_xy, angle, is_start, local_radius, z_center, settings, idx):
    tangent = seam_tangent(angle)
    normal_into_part = tangent if is_start else -tangent
    radial = Vector((math.cos(angle), math.sin(angle), 0.0))
    cx, cy = center_xy

    r0 = max(0.0, local_radius - settings.voxel_size)
    r1 = local_radius + settings.wing_width
    half_span = settings.wing_span / 2.0

    base = Vector((cx, cy, z_center)) + radial * r0
    far = Vector((cx, cy, z_center)) + radial * r1

    mesh = bpy.data.meshes.new(f"Wing_{idx}")
    obj = bpy.data.objects.new(f"Wing_{idx}", mesh)
    coll.objects.link(obj)
    bm = bmesh.new()
    verts = []
    for r_pt in (base, far):
        for n_off in (0.0, settings.wing_thickness):
            for z_off in (-half_span, half_span):
                verts.append(bm.verts.new(r_pt + normal_into_part * n_off + Vector((0, 0, z_off))))
    # verts order: [r0,n0,z-][r0,n0,z+][r0,n1,z-][r0,n1,z+][r1,n0,z-][r1,n0,z+][r1,n1,z-][r1,n1,z+]
    v = verts
    faces = [
        (v[0], v[1], v[3], v[2]),  # inner (radial r0) face
        (v[4], v[6], v[7], v[5]),  # outer (radial r1) face
        (v[0], v[4], v[5], v[1]),  # normal=0 face
        (v[2], v[3], v[7], v[6]),  # normal=thickness face
        (v[0], v[2], v[6], v[4]),  # bottom
        (v[1], v[5], v[7], v[3]),  # top
    ]
    for f in faces:
        bm.faces.new(f)
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
    bm.to_mesh(mesh)
    bm.free()

    part_obj = boolean_combine(part_obj, obj, 'UNION', delete_b=True)

    if settings.wing_hole_diameter > 1e-9:
        hole_center = Vector((cx, cy, z_center)) + radial * ((r0 + r1) / 2.0) + normal_into_part * (settings.wing_thickness / 2.0)
        hole = make_cylinder(
            f"WingHole_{idx}", hole_center, settings.wing_hole_diameter / 2.0,
            settings.wing_thickness * 3.0, normal_into_part, coll, segments=16,
        )
        part_obj = boolean_combine(part_obj, hole, 'DIFFERENCE', delete_b=True)

    return part_obj


def add_lock(part_obj, coll, center_xy, angle, local_radius, wall_thickness, z_center, settings, is_male, idx):
    radial = Vector((math.cos(angle), math.sin(angle), 0.0))
    cx, cy = center_xy
    r_mid = local_radius + wall_thickness / 2.0
    center = Vector((cx, cy, z_center)) + radial * r_mid

    if is_male:
        bump = make_sphere(f"LockM_{idx}", center, settings.lock_diameter / 2.0, coll)
        part_obj = boolean_combine(part_obj, bump, 'UNION', delete_b=True)
    else:
        socket = make_sphere(f"LockF_{idx}", center, settings.lock_diameter / 2.0 + settings.lock_clearance, coll)
        part_obj = boolean_combine(part_obj, socket, 'DIFFERENCE', delete_b=True)

    return part_obj


class MOLDJACKET_OT_generate_jacket(Operator):
    bl_idname = "moldjacket.generate_jacket"
    bl_label = "Generate Jacket"
    bl_description = (
        "Build the split mold jacket: an offset silicone gap, a rigid wall shell, "
        "trimmed top/bottom, split into parts with wings and mama/papa alignment locks"
    )
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        s = context.scene.mold_jacket
        return working_master(s) is not None

    def execute(self, context):
        settings = context.scene.mold_jacket
        master = working_master(settings)
        coll = get_or_create_collection(f"MoldJacket - {settings.master_object.name}")

        coords = world_vert_coords(master)
        minb, maxb = world_bounds(master)
        cx = (minb[0] + maxb[0]) / 2.0
        cy = (minb[1] + maxb[1]) / 2.0
        center_xy = (cx, cy)
        base_z = minb[2]
        top_z = maxb[2] + settings.top_margin

        gap = settings.gap_distance
        wall = settings.wall_thickness
        vsize = settings.voxel_size

        self.report({'INFO'}, "Building offset cavity surface...")
        cavity = offset_filled_solid(master, gap, vsize, "Cavity_tmp", coll)
        self.report({'INFO'}, "Building offset wall surface...")
        outer = offset_filled_solid(master, gap + wall, vsize, "Outer_tmp", coll)
        jacket = boolean_combine(outer, cavity, 'DIFFERENCE', delete_b=True)
        # Chained EXACT booleans are fragile: a handful of leftover degenerate
        # faces here will make the *next* boolean silently fail. Clean up now.
        clean_solid(jacket, vsize)

        # cutters only need to comfortably exceed the model footprint - making
        # them huge relative to a small (cm-scale) model starves the exact
        # boolean solver of precision and corrupts the result.
        model_span = max(maxb[0] - minb[0], maxb[1] - minb[1], maxb[2] - minb[2])
        big = model_span * 4.0 + gap + wall + vsize * 4.0

        # trim to [base_z, top_z] - open top, sits on the base plate at the bottom
        trim_box = make_box_from_bounds(
            "TrimBox", cx - big, cx + big, cy - big, cy + big, base_z, top_z, coll,
        )
        jacket = boolean_combine(jacket, trim_box, 'INTERSECT', delete_b=True)
        clean_solid(jacket, vsize)

        # subtract holes for the base-plate alignment pins
        _, plate_base_z, _model_radius, pin_positions = compute_pin_positions(settings, master)
        for i, (px, py, _ang) in enumerate(pin_positions):
            hole = make_cylinder(
                f"PinHole_{i}",
                Vector((px, py, base_z)),
                settings.pin_diameter / 2.0 + settings.lock_clearance,
                settings.pin_height * 3.0,
                Vector((0, 0, 1)), coll, segments=16,
            )
            jacket = boolean_combine(jacket, hole, 'DIFFERENCE', delete_b=True)
        if settings.pin_count > 0:
            clean_solid(jacket, vsize)

        # split into N parts around Z through the model's XY center
        n = settings.num_parts
        offset = settings.split_angle_offset
        boundary_angles = [offset + 2 * math.pi * i / n for i in range(n + 1)]

        base_mesh = jacket.data
        parts = []
        for i in range(n):
            part_mesh = base_mesh.copy()
            part_obj = bpy.data.objects.new(f"{settings.master_object.name}_JacketPart{i + 1}", part_mesh)
            coll.objects.link(part_obj)
            z_pad = (top_z - base_z) * 0.5 + vsize * 4.0
            wedge = make_wedge_prism(
                f"Wedge_{i}", center_xy, boundary_angles[i], boundary_angles[i + 1],
                big, base_z - z_pad, top_z + z_pad, coll,
            )
            part_obj = boolean_combine(part_obj, wedge, 'INTERSECT', delete_b=True)
            clean_solid(part_obj, vsize)
            parts.append(part_obj)
        remove_object(jacket)

        # wings + locks on each seam (each part has a "start" boundary a0 and
        # an "end" boundary a1; the end side of part i is the male/papa side,
        # matching the start side of part i+1, which is female/mama)
        for i, part_obj in enumerate(parts):
            a0, a1 = boundary_angles[i], boundary_angles[i + 1]
            for boundary_angle, is_start in ((a0, True), (a1, False)):
                for w in range(settings.wings_per_seam):
                    frac = (w + 1) / (settings.wings_per_seam + 1)
                    z_center = base_z + (top_z - base_z) * frac
                    local_r = footprint_radius(
                        coords, center_xy,
                        z_min=z_center - settings.wing_span, z_max=z_center + settings.wing_span,
                    ) + gap + wall
                    part_obj = add_wing(
                        part_obj, coll, center_xy, boundary_angle, is_start,
                        local_r, z_center, settings, f"{i}_{boundary_angle:.2f}_{w}",
                    )
                is_male = not is_start
                for k in range(settings.locks_per_seam):
                    frac = (k + 1) / (settings.locks_per_seam + 1)
                    z_center = base_z + (top_z - base_z) * frac
                    local_r = footprint_radius(
                        coords, center_xy,
                        z_min=z_center - 0.01, z_max=z_center + 0.01,
                    ) + gap
                    part_obj = add_lock(
                        part_obj, coll, center_xy, boundary_angle, local_r, wall,
                        z_center, settings, is_male, f"{i}_{boundary_angle:.2f}_{k}",
                    )
            recalc_normals(part_obj)
            parts[i] = part_obj

        self.report({'INFO'}, f"Jacket generated: {n} parts in collection '{coll.name}'")
        return {'FINISHED'}


# ---------------------------------------------------------------------------
# Operator: Generate All (prepare + base plate + jacket in one click)
# ---------------------------------------------------------------------------

class MOLDJACKET_OT_generate_all(Operator):
    bl_idname = "moldjacket.generate_all"
    bl_label = "Generate All"
    bl_description = "Run Cut Bottom + Generate Base Plate + Generate Jacket in sequence"
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        s = context.scene.mold_jacket
        return s.master_object is not None and s.master_object.type == 'MESH'

    def execute(self, context):
        bpy.ops.moldjacket.prepare_master()
        bpy.ops.moldjacket.generate_base_plate()
        bpy.ops.moldjacket.generate_jacket()
        return {'FINISHED'}


# ---------------------------------------------------------------------------
# UI Panel
# ---------------------------------------------------------------------------

class MOLDJACKET_PT_main(Panel):
    bl_label = "Mold Jacket / Опалубка"
    bl_idname = "MOLDJACKET_PT_main"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "Mold Jacket"

    def draw(self, context):
        layout = self.layout
        settings = context.scene.mold_jacket

        layout.prop(settings, "master_object")

        box = layout.box()
        box.label(text="1. Prepare Master (cut open bottom)")
        row = box.row(align=True)
        row.prop(settings, "cut_z")
        row.operator("moldjacket.set_cut_to_min", text="", icon='TRIA_DOWN_BAR')
        box.operator("moldjacket.prepare_master")
        if settings.prepared_object:
            box.label(text=f"Prepared: {settings.prepared_object.name}", icon='CHECKMARK')

        box = layout.box()
        box.label(text="2. Base Plate")
        box.prop(settings, "base_margin")
        box.prop(settings, "base_plate_thickness")
        row = box.row(align=True)
        row.prop(settings, "step1_height")
        row.prop(settings, "step1_width")
        row = box.row(align=True)
        row.prop(settings, "step2_height")
        row.prop(settings, "step2_width")
        row = box.row(align=True)
        row.prop(settings, "pin_count")
        row.prop(settings, "pin_diameter")
        box.prop(settings, "pin_height")
        box.operator("moldjacket.generate_base_plate")

        box = layout.box()
        box.label(text="3. Mold Jacket (опалубка)")
        box.prop(settings, "gap_distance")
        box.prop(settings, "wall_thickness")
        box.prop(settings, "voxel_size")
        box.prop(settings, "top_margin")
        box.prop(settings, "num_parts")
        box.prop(settings, "split_angle_offset")

        sub = box.box()
        sub.label(text="Wings")
        sub.prop(settings, "wings_per_seam")
        sub.prop(settings, "wing_width")
        sub.prop(settings, "wing_span")
        sub.prop(settings, "wing_thickness")
        sub.prop(settings, "wing_hole_diameter")

        sub = box.box()
        sub.label(text="Mama/Papa Locks")
        sub.prop(settings, "locks_per_seam")
        sub.prop(settings, "lock_diameter")
        sub.prop(settings, "lock_depth")
        sub.prop(settings, "lock_clearance")

        box.operator("moldjacket.generate_jacket")

        layout.separator()
        layout.operator("moldjacket.generate_all", icon='PLAY')


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------

classes = (
    MoldJacketSettings,
    MOLDJACKET_OT_prepare_master,
    MOLDJACKET_OT_set_cut_to_min,
    MOLDJACKET_OT_generate_base_plate,
    MOLDJACKET_OT_generate_jacket,
    MOLDJACKET_OT_generate_all,
    MOLDJACKET_PT_main,
)


def register():
    for cls in classes:
        bpy.utils.register_class(cls)
    bpy.types.Scene.mold_jacket = PointerProperty(type=MoldJacketSettings)


def unregister():
    del bpy.types.Scene.mold_jacket
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)


if __name__ == "__main__":
    register()
