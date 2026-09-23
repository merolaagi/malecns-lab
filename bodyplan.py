"""Leg joints driven by the motor neurons that innervate their muscles.

MaleCNS motor neuron types name the muscle each one drives ("Ti flexor MN", "Sternal anterior rotator
MN"), so they can be grouped into the antagonist pairs of each leg joint instead of pooled into a single
number per leg. That is what this module does, for the six legs:

    coxa    protraction  <- Sternal anterior rotator, Tergopleural/Pleural promotor
            retraction   <- Sternal posterior rotator, Pleural remotor/abductor, Sternal adductor
    femur   levation     <- Tr extensor
            depression   <- Tr flexor, Acc. tr flexor, Sternotrochanter, Tergotrochanter
    tibia   extension    <- Ti extensor
            flexion      <- Ti flexor, Acc. ti flexor
    tarsus  levation     <- Ta levator
            depression   <- Ta depressor, long tendon muscle (ltm)

Measured: which cells belong to which muscle, and their firing rates in the simulation.
Assumed: that a joint angle follows the difference in firing rate between its antagonists through
    angle = rest + span * tanh((rate_agonist - rate_antagonist) / scale),
the rest angles, the spans, the scale, and the segment lengths used for drawing. There is no muscle
model, no tendon, no load and no inertia, so these are readouts of motor activity in the shape of a
leg, not a biomechanical simulation.
"""
import numpy as np

MUSCLES = {
    'coxa_protract': ['Sternal anterior rotator', 'Tergopleural/Pleural promotor'],
    'coxa_retract': ['Sternal posterior rotator', 'Pleural remotor/abductor', 'Sternal adductor'],
    'femur_levate': ['Tr extensor'],
    'femur_depress': ['Tr flexor', 'Acc. tr flexor', 'Sternotrochanter', 'Tergotr.'],
    'tibia_extend': ['Ti extensor'],
    'tibia_flex': ['Ti flexor', 'Acc. ti flexor'],
    'tarsus_levate': ['Ta levator'],
    'tarsus_depress': ['Ta depressor', 'ltm', 'ltm1-tibia', 'ltm2-femur'],
}
# joint: (agonist group, antagonist group, rest angle in radians, span in radians)
JOINTS = {
    'coxa': ('coxa_protract', 'coxa_retract', 0.0, 0.55),      # fore-aft swing, seen from above
    'femur': ('femur_levate', 'femur_depress', 0.0, 1.0),      # positive lifts the leg off the ground
    'tibia': ('tibia_extend', 'tibia_flex', 0.6, 0.6),         # knee angle; larger reaches further
    'tarsus': ('tarsus_levate', 'tarsus_depress', 0.0, 0.6),
}
LEGS = ['LF', 'LM', 'LH', 'RF', 'RM', 'RH']
SEGMENTS = {'coxa': 0.30, 'femur': 0.75, 'tibia': 0.75, 'tarsus': 0.55}   # mm, approximate
RATE_SCALE = 12.0          # firing-rate difference (Hz) that saturates a joint


def muscle_of(type_name):
    if not type_name: return None
    base = str(type_name).replace(' MN', '').strip()
    for group, names in MUSCLES.items():
        if base in names: return group
    return None


def leg_of(node):
    side = node.get('rootSide') or node.get('somaSide')
    part = {'fl': 'F', 'ml': 'M', 'hl': 'H'}.get(node.get('subclass'))
    label = f'{side}{part}'
    return label if label in LEGS else None


def index(nodes):
    """{leg: {muscle group: array of cell indices}} plus the unmatched motor neurons."""
    groups = {leg: {g: [] for g in MUSCLES} for leg in LEGS}
    unmatched = []
    for i, n in enumerate(nodes):
        if n.get('superclass') != 'vnc_motor': continue
        leg, muscle = leg_of(n), muscle_of(n.get('type'))
        if leg and muscle: groups[leg][muscle].append(i)
        else: unmatched.append(i)
    return {'groups': {leg: {g: np.array(v, dtype=int) for g, v in m.items()} for leg, m in groups.items()},
            'unmatched': np.array(unmatched, dtype=int)}


def coverage(idx):
    """Which joints each leg can actually drive, for honest reporting."""
    out = {}
    for leg, m in idx['groups'].items():
        out[leg] = {joint: {'agonist': int(len(m[a])), 'antagonist': int(len(m[b])),
                            'drivable': bool(len(m[a]) and len(m[b]))}
                    for joint, (a, b, _, _) in JOINTS.items()}
    return out


def angles(idx, rates):
    """Joint angles per leg from the current firing rates (Hz) of each muscle's motor neurons."""
    out = {}
    for leg, m in idx['groups'].items():
        joint_angles = {}
        for joint, (a, b, rest, span) in JOINTS.items():
            ra = float(rates[m[a]].mean()) if len(m[a]) else 0.0
            rb = float(rates[m[b]].mean()) if len(m[b]) else 0.0
            joint_angles[joint] = float(rest + span * np.tanh((ra - rb) / RATE_SCALE)) if (len(m[a]) or len(m[b])) else rest
            joint_angles[joint + '_hz'] = [round(ra, 2), round(rb, 2)]
        out[leg] = joint_angles
    return out


def foot(leg, joint_angles):
    """Foot position in the body frame (mm), seen from above, and whether the leg is planted.

    Seen from above, the coxa angle swings the leg fore and aft, tibia extension sets how far it
    reaches, and femur levation lifts it; a lifted leg is in swing and carries no load."""
    side = -1 if leg[0] == 'L' else 1
    row = {'F': 0.55, 'M': 0.0, 'H': -0.55}[leg[1]]
    lift = joint_angles['femur']
    reach = SEGMENTS['coxa'] + (SEGMENTS['femur'] + SEGMENTS['tibia']) * (0.45 + 0.35 * np.cos(joint_angles['tibia'])) + SEGMENTS['tarsus'] * 0.5
    reach *= max(0.2, 1 - 0.35 * max(lift, 0.0))              # a lifted leg projects shorter from above
    swing = joint_angles['coxa']
    x = row + reach * np.sin(swing)
    y = side * reach * np.cos(swing) * 0.75
    return float(x), float(y), bool(lift <= 0)                 # planted when the femur is depressed
