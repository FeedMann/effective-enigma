"""Pure-python reader for X-Ray engine binary formats (OGF / OMF / OMF-banks).

Format reference: blender_stalker_tools_addon (PavelBlend) + io_scene_xray chunk ids.
No Blender, no game engine needed.
"""
import struct

# ---------------------------------------------------------------- chunk ids
C_HEADER = 0x1
C_TEXTURE = 0x2
C_VERTICES = 0x3
C_INDICES = 0x4
C_SWIDATA = 0x6
C_VCONTAINER = 0x7
C_ICONTAINER = 0x8
C_CHILDREN = 0x9
C_CHILDREN_L = 0xA
C_S_BONE_NAMES = 0xD
C_S_MOTIONS = 0xE
C_S_SMPARAMS = 0xF
C_S_IKDATA = 0x10
C_S_USERDATA = 0x11
C_DESC = 0x12
C_S_MOTION_REFS = 0x13
C_SWICONTAINER = 0x14
C_GCONTAINER = 0x15
C_FASTPATH = 0x16
# version-3 variants
C3_S_SMPARAMS_NEW = 0x14
C3_S_MOTIONS_NEW = 0x1A

FL_T_KEY_PRESENT = 1 << 0
FL_R_KEY_ABSENT = 1 << 1
KPF_T_HQ = 1 << 2

MODEL_TYPES = {
    0x0: 'NORMAL', 0x1: 'HIERRARHY', 0x2: 'PROGRESSIVE', 0x3: 'SKELETON_ANIM',
    0x4: 'SKELETON_GEOMDEF_PM', 0x5: 'SKELETON_GEOMDEF_ST', 0x6: 'LOD',
    0x7: 'TREE_ST', 0x8: 'PARTICLE_EFFECT', 0x9: 'PARTICLE_GROUP',
    0xA: 'SKELETON_RIGID', 0xB: 'TREE_PM',
}
VF_1L = 0x12071980
VF_2L = 0x240E3300
VF_NL = 0x36154C80


class CompressedChunk(NotImplementedError):
    pass


class Chunk:
    __slots__ = ('id', 'data', 'compressed')

    def __init__(self, cid, data, compressed=False):
        self.id = cid
        self.data = data
        self.compressed = compressed

    def __repr__(self):
        return '<Chunk 0x%X len=%d%s>' % (self.id, len(self.data), ' COMPRESSED' if self.compressed else '')


def lzo1x_decompress(src, dst_len):
    """Minimal LZO1X decompressor (x-ray 'ogz' chunk payloads)."""
    ip, op = 0, 0
    out = bytearray()
    n = len(src)
    t = 0
    if src[0] > 17:
        t = src[0] - 17
        out += src[1:1 + t]
        ip = 1 + t
        if t < 4:
            raise ValueError('lzo: bad initial literal run')
    while True:
        if not (src[ip] & 0x80):
            # short match
            t = src[ip] & 0x0F
            ip += 1
            if t == 0:
                # long literal run
                t = 15
                while src[ip] == 0:
                    t += 255
                    ip += 1
                t += src[ip]
                ip += 1
                out += src[ip:ip + t + 3]
                ip += t + 3
                # next: is it a match or EOF?
                if ip >= n:
                    break
                continue
            # match found -> goto copy
            m_pos = len(out) - 1 - ((src[ip] >> 2) if ip < n else 0)
            m_len = t + 2
            ip += 1
        else:
            t = src[ip] & 0x1F
            ip += 1
            if t == 0:
                t = 31
                while src[ip] == 0:
                    t += 255
                    ip += 1
                t += src[ip]
                ip += 1
            m_len = t + 2
            if t >= 8:
                m_pos = len(out) - 1 - ((src[ip] >> 2) + (src[ip + 1] << 6) if False else 0)
            # handled below in simple safe variant
            raise NotImplementedError('lzo1x writer fallback')
        # literal run following
        break
    raise NotImplementedError('lzo1x')


def lzo_decompress_span(src, out_len):
    """Straight port of the reference lzo1x_decompress_safe core loop."""
    out = bytearray()
    ip = 0
    n = len(src)

    def copy_match(m_pos, m_len):
        for _ in range(m_len):
            out.append(out[m_pos])
            m_pos += 1

    if n == 0:
        return bytes(out)
    t = 0
    if src[ip] > 17:
        t = src[ip] - 17
        ip += 1
        out += src[ip:ip + t]
        ip += t
        if t < 4:
            return _lzo_loop(src, ip, out, n)
    return _lzo_loop(src, ip, out, n)


def _lzo_loop(src, ip, out, n):
    while True:
        if ip >= n:
            break
        b = src[ip]
        if b >= 16:  # 0x40..0xFF : long match
            t = b & 0x1F
            ip += 1
            if t == 0:
                t = 31
                while src[ip] == 0:
                    t += 255
                    ip += 1
                t += src[ip]
                ip += 1
            m_len = t + 2
            if b < 0x40:
                m_pos = len(out) - 1 - (src[ip] >> 2)
                ip += 1
            else:
                m_pos = len(out) - 1 - (((src[ip] >> 2) + (src[ip + 1] << 6)))
                ip += 2
            for _ in range(m_len):
                out.append(out[m_pos])
                m_pos += 1
            # then a literal run always follows in lzo1x
            t = src[ip] + 3
            ip += 1
            out += src[ip:ip + t]
            ip += t
        elif b < 16:  # 0x00..0x0F : short literal-run + match
            t = b
            ip += 1
            if t == 0:
                while src[ip] == 0:
                    t += 255
                    ip += 1
                t += 15 + src[ip]
                ip += 1
            out += src[ip:ip + t + 3]
            ip += t + 3
            m_len = src[ip] >> 2
            ip += 1
            if m_len == 0:
                m_len = 1
                while src[ip] == 0:
                    m_len += 255
                    ip += 1
                m_len += src[ip]
                ip += 1
            m_len += 2
            m_pos = len(out) - 1 - (src[ip] >> 2)
            ip += 1
            if ip - 1 < n and False:
                pass
            for _ in range(m_len):
                out.append(out[m_pos])
                m_pos += 1
        else:  # 0x10..0x3F : rare literal run, no match
            t = b & 0x0F
            ip += 1
            if t == 0:
                while src[ip] == 0:
                    t += 255
                    ip += 1
                t += 15 + src[ip]
                ip += 1
            out += src[ip:ip + t + 3]
            ip += t + 3
            if ip >= n:
                break
    return bytes(out)


def read_chunks(data, offset=0, end=None):
    """Yield Chunk objects from a buffer."""
    end = len(data) if end is None else end
    out = []
    while offset < end:
        if offset + 8 > end:
            raise ValueError('truncated chunk header at %d/%d' % (offset, end))
        cid, size = struct.unpack_from('<II', data, offset)
        offset += 8
        compressed = bool(size & 0x80000000)
        if compressed:
            size &= 0x7FFFFFFF
        payload = data[offset:offset + size]
        if len(payload) != size:
            raise ValueError('truncated chunk 0x%X at %d' % (cid, offset))
        offset += size
        if compressed:
            real = struct.unpack_from('<I', payload, 0)[0]
            payload = lzo_decompress_span(payload[4:], real)
            if len(payload) != real:
                raise ValueError('lzo: got %d want %d' % (len(payload), real))
        out.append(Chunk(cid, payload, compressed))
    return out


class PackedReader:
    __slots__ = ('data', 'pos')

    def __init__(self, data):
        self.data = data
        self.pos = 0

    def unpack(self, fmt):
        size = struct.calcsize(fmt)
        vals = struct.unpack_from(fmt, self.data, self.pos)
        self.pos += size
        return vals

    def getf(self, fmt):
        if len(fmt) > 1 and isinstance(fmt, str) and fmt[0].isdigit():
            return self.unpack('<%s' % fmt)
        return self.unpack('<' + fmt)

    def gets(self):
        """null-terminated string (cp1251, x-ray convention)"""
        i = self.data.index(b'\0', self.pos)
        s = self.data[self.pos:i]
        self.pos = i + 1
        try:
            return s.decode('cp1251')
        except UnicodeDecodeError:
            return s.decode('latin1')


# ---------------------------------------------------------------- OMF / motions
class BoneMotion:
    __slots__ = ('name', 't_present', 'r_absent', 'hq', 'quats', 'translate', 't_size', 't_init')

    def __init__(self, name):
        self.name = name
        self.t_present = False
        self.r_absent = False
        self.hq = False
        self.quats = []       # list of (x, y, z, w) int16 raw
        self.translate = []   # list of (x, y, z)
        self.t_size = (1.0, 1.0, 1.0)
        self.t_init = (0.0, 0.0, 0.0)


class Motion:
    __slots__ = ('name', 'length', 'bones', 'bone_order')

    def __init__(self, name, length):
        self.name = name
        self.length = length
        self.bones = {}
        self.bone_order = []

    def __repr__(self):
        return '<Motion %r frames=%d bones=%d>' % (self.name, self.length, len(self.bones))


def read_bone_motion(pr, motion, bone_name):
    bm = BoneMotion(bone_name)
    flags = pr.getf('B')[0]
    bm.t_present = bool(flags & FL_T_KEY_PRESENT)
    bm.r_absent = bool(flags & FL_R_KEY_ABSENT)
    bm.hq = bool(flags & KPF_T_HQ)

    if bm.r_absent:
        bm.quats.append(pr.getf('4h'))
    else:
        pr.getf('I')  # crc32 of the motion name
        for _ in range(motion.length):
            bm.quats.append(pr.getf('4h'))

    if bm.t_present:
        pr.getf('I')  # crc32
        fmt = '3h' if bm.hq else '3b'
        for _ in range(motion.length):
            bm.translate.append(pr.getf(fmt))
        bm.t_size = pr.getf('3f')
        bm.t_init = pr.getf('3f')
    else:
        bm.translate.append(pr.getf('3f'))
    return bm


def read_motion(chunk_data, bone_names):
    pr = PackedReader(chunk_data)
    name = pr.gets()
    length = pr.getf('I')[0]
    motion = Motion(name, length)
    for bone_name in bone_names:
        motion.bones[bone_name] = read_bone_motion(pr, motion, bone_name)
        motion.bone_order.append(bone_name)
    if pr.pos != len(chunk_data):
        raise ValueError('motion %r: consumed %d of %d bytes' % (name, pr.pos, len(chunk_data)))
    return motion


def read_motions(data, bone_names):
    """S_MOTIONS / S_MOTIONS_NEW chunk -> dict name->Motion, plus declared count."""
    motions = {}
    declared = None
    for chunk in read_chunks(data):
        if chunk.id == 0:
            declared = struct.unpack_from('<I', chunk.data, 0)[0]
            continue
        m = read_motion(chunk.data, bone_names)
        motions[m.name] = m
    return motions, declared


def read_smparams(data, verbose=False):
    """S_SMPARAMS -> (version, partitions, motion_defs, marks)"""
    pr = PackedReader(data)
    version = pr.getf('H')[0]
    n_parts = pr.getf('H')[0]
    partitions = []
    for _ in range(n_parts):
        pname = pr.gets()
        n_bones = pr.getf('H')[0]
        names, ids = [], []
        for _ in range(n_bones):
            if version == 1:
                ids.append(pr.getf('I')[0])
            elif version == 2:
                names.append(pr.gets())
            elif version in (3, 4):
                names.append(pr.gets())
                ids.append(pr.getf('I')[0])
            else:
                raise ValueError('unknown smparams version %d' % version)
        partitions.append({'name': pname, 'names': names, 'ids': ids})
    n_motions = pr.getf('H')[0]
    motions = []
    for _ in range(n_motions):
        mname = pr.gets()
        flags = pr.getf('I')[0]
        bone_or_part = pr.getf('H')[0]
        motion = pr.getf('H')[0]
        speed, power, accrue, falloff = pr.getf('4f')
        marks = []
        if version == 4:
            n_marks = pr.getf('I')[0]
            for _ in range(n_marks):
                mtype = pr.getf('B')[0]
                mpname = pr.gets()
                mval = pr.getf('f')[0]
                marks.append((mtype, mpname, mval))
        motions.append({'name': mname, 'flags': flags, 'partition_or_bone': bone_or_part,
                        'motion_id': motion, 'speed': speed, 'power': power,
                        'accrue': accrue, 'falloff': falloff, 'marks': marks})
    trailing = len(data) - pr.pos
    return {'version': version, 'partitions': partitions, 'motions': motions,
            'trailing_bytes': trailing}
