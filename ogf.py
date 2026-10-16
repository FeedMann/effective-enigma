"""OGF (X-Ray model) reader - pure python port of the plugin conventions."""
import struct
import numpy as np
import xray


VF_FVF = 0x112
VF_1L = 0x12071980
VF_2L = 0x240E3300
VF_NL = 0x36154C80


def _obb(pr):
    rot = pr.getf('9f')
    trans = pr.getf('3f')
    half = pr.getf('3f')
    return {'rotate': rot, 'translate': trans, 'halfsize': half}


def read_bone_names(data):
    pr = xray.PackedReader(data)
    count = pr.getf('I')[0]
    bones = []
    for _ in range(count):
        name = pr.gets()
        parent = pr.gets()
        bones.append({'name': name, 'parent': parent, 'obb': _obb(pr)})
    return bones


def read_ikdata(data, bone_names):
    pr = xray.PackedReader(data)
    bones = []
    for name, parent in bone_names:
        version = pr.getf('I')[0]
        game_material = pr.gets()
        shape_type = pr.getf('H')[0]
        shapes = []
        if shape_type == 0x1:  # sphere
            shapes.append(pr.getf('4f'))
        elif shape_type == 0x2:  # box
            shapes.append(pr.getf('9f'))
        elif shape_type == 0x3:  # cylinder
            shapes.append(pr.getf('8f'))
            _obb(pr)
        for _ in range(3):
            pr.getf('2f'); pr.getf('f'); pr.getf('f')
        pr.getf('f'); pr.getf('f')
        pr.getf('I')
        pr.getf('f'); pr.getf('f'); pr.getf('f')
        rotate_x, rotate_y, rotate_z = pr.getf('3f')
        ox, oy, oz = pr.getf('3f')
        mass = pr.getf('f')[0]
        com = pr.getf('3f')
        bones.append({
            'name': name, 'parent': parent, 'game_material': game_material,
            'rotate_file': (rotate_x, rotate_y, rotate_z),
            'offset_file': (ox, oy, oz), 'mass': mass, 'center_of_mass': com,
        })
    return bones


def read_texture(data):
    pr = xray.PackedReader(data)
    return {'texture': pr.gets(), 'shader': pr.gets()}


def read_vertices(data):
    pr = xray.PackedReader(data)
    fmt = pr.getf('I')[0]
    count = pr.getf('I')[0]
    verts, uvs, normals, weights, used = [], [], [], [], set()
    if fmt == VF_FVF:
        for _ in range(count):
            p = pr.getf('3f'); n = pr.getf('3f'); uv = pr.getf('2f')
            verts.append((p[0], p[2], p[1])); normals.append((n[0], n[2], n[1]))
            uvs.append((uv[0], 1 - uv[1])); weights.append([])
    elif fmt == VF_1L:
        for _ in range(count):
            p = pr.getf('3f'); n = pr.getf('3f'); pr.getf('6f'); uv = pr.getf('2f')
            bone = pr.getf('I')[0]
            used.add(bone)
            verts.append((p[0], p[2], p[1])); normals.append((n[0], n[2], n[1]))
            uvs.append((uv[0], 1 - uv[1])); weights.append([(bone, 1.0)])
    elif fmt == VF_2L:
        for _ in range(count):
            b0, b1 = pr.getf('2H')
            p = pr.getf('3f'); n = pr.getf('3f'); pr.getf('6f')
            infl = pr.getf('f')[0]; uv = pr.getf('2f')
            used.add(b0); used.add(b1)
            verts.append((p[0], p[2], p[1])); normals.append((n[0], n[2], n[1]))
            uvs.append((uv[0], 1 - uv[1]))
            if b0 == b1:
                weights.append([(b0, 1.0)])
            else:
                weights.append([(b0, 1.0 - infl), (b1, infl)])
    else:
        raise ValueError('unsupported vertex format 0x%X' % fmt)
    return {'format': fmt, 'verts': verts, 'normals': normals, 'uvs': uvs,
            'weights': weights, 'used_bones': used}


def read_indices(data):
    pr = xray.PackedReader(data)
    count = pr.getf('I')[0]
    return [pr.getf('H')[0] for _ in range(count)]


def read_swidata(data):
    pr = xray.PackedReader(data)
    pr.getf('4I')
    count = pr.getf('I')[0]
    swis = []
    for _ in range(count):
        swis.append({'offset': pr.getf('I')[0], 'tris': pr.getf('H')[0], 'verts': pr.getf('H')[0]})
    return swis


def read_visual(data, child=False, depth=0):
    """Parse one visual's chunk set (recursively)."""
    vis = {'children': [], 'chunks': []}
    for c in xray.read_chunks(data):
        vis['chunks'].append(hex(c.id))
        if c.id == xray.C_HEADER:
            pr = xray.PackedReader(c.data)
            vis['version'] = pr.getf('B')[0]
            vis['type_id'] = pr.getf('B')[0]
            vis['type'] = xray.MODEL_TYPES.get(vis['type_id'], 'UNKNOWN_%d' % vis['type_id'])
            vis['shader_id'] = pr.getf('H')[0]
            if vis['version'] == 4:
                vis['bbox'] = (pr.getf('3f'), pr.getf('3f'))
                vis['bsphere'] = (pr.getf('3f'), pr.getf('f')[0])
        elif c.id == xray.C_S_BONE_NAMES:
            vis['bone_names'] = read_bone_names(c.data)
        elif c.id == xray.C_S_IKDATA:
            pairs = [(b['name'], b['parent']) for b in vis.get('bone_names', [])]
            vis['bones'] = read_ikdata(c.data, pairs)
        elif c.id == xray.C_S_SMPARAMS:
            vis['smparams'] = xray.read_smparams(c.data)
        elif c.id == xray.C_TEXTURE:
            vis.update(read_texture(c.data))
        elif c.id == xray.C_VERTICES:
            vis['mesh'] = read_vertices(c.data)
        elif c.id == xray.C_INDICES:
            vis['indices'] = read_indices(c.data)
        elif c.id == xray.C_SWIDATA:
            vis['swidata'] = read_swidata(c.data)
        elif c.id == xray.C_VCONTAINER:
            pr = xray.PackedReader(c.data)
            vis['vcontainer'] = pr.getf('3I')
        elif c.id == xray.C_ICONTAINER:
            pr = xray.PackedReader(c.data)
            vis['icontainer'] = pr.getf('3I')
        elif c.id == xray.C_CHILDREN:
            for sub in xray.read_chunks(c.data):
                vis['children'].append(read_visual(sub.data, child=True, depth=depth + 1))
        elif c.id == xray.C_CHILDREN_L:
            pr = xray.PackedReader(c.data)
            vis['children_l'] = [pr.getf('I')[0] for _ in range(pr.getf('I')[0])]
        elif c.id in (xray.C_S_USERDATA, xray.C_DESC, xray.C_FASTPATH, xray.C_S_MOTION_REFS, xray.C_GCONTAINER, xray.C_SWICONTAINER):
            pass
        else:
            vis.setdefault('unknown', []).append(hex(c.id))
    return vis


def read_ogf(path):
    data = open(path, 'rb').read()
    return read_visual(data)


def bone_chain(ogf_bones):
    """Return world matrices for the OGF rest skeleton using plugin conventions.
    plugin: offset = (ox, oz, oy) ; rotate euler(x)=-rx, y=-rz, z=-ry with order 'YXZ'
    """
    import smd
    idx = {b['name']: i for i, b in enumerate(ogf_bones)}
    locals_ = []
    for b in ogf_bones:
        ox, oy, oz = b['offset_file']
        pos = np.array([ox, oz, oy], dtype=np.float64)
        rx, ry, rz = b['rotate_file']
        R = smd.euler_matrix_yxz(-rx, -rz, -ry)
        m = np.eye(4)
        m[:3, :3] = R
        m[:3, 3] = pos
        locals_.append(m)
    worlds = []
    for i, b in enumerate(ogf_bones):
        p = idx.get(b['parent'], -1)
        worlds.append(worlds[p] @ locals_[i] if p >= 0 else locals_[i])
    return locals_, worlds
