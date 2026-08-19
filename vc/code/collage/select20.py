#!/usr/bin/env python
"""Pick 20 trajectories spanning the emotion space, including 6 mirror pairs (both directions)."""
import json, numpy as np, collections, os

NB = '/e/data1/datasets/playground/mmlaion/schuhmann1/dramabox'
CT = NB + '/collage_traj'
W = NB + '/vc4arm'

C = {x['id']: x for x in json.load(open(CT + '/collages.json'))}
REGION = np.load(CT + '/region.npy')
pm = json.load(open(CT + '/pathmap.json'))
Z = np.load(CT + '/graph.npz')['Zst']

FAM = lambda v: ('anime' if v.startswith('anime') else 'emolia' if v.startswith('emolia')
                 else 'mediathek' if v.startswith('mediathek') else 'refvoice' if v.startswith('refvoice')
                 else 'k')

# 6 mirror emotion pairs, both directions.  4 pairs from the conservative op and
# 2 from the relaxed op, so the selection also spans both path-threshold settings.
MIRROR_IDS = ['C000', 'C001',   # Anger <-> Thankfulness_Gratitude   (conservative)
              'C002', 'C003',   # Sadness <-> Elation                (conservative)
              'C006', 'C007',   # Fear <-> Pride                     (conservative)
              'C010', 'C011',   # Amusement <-> Distress             (conservative)
              'C054', 'C055',   # Contempt <-> Affection             (relaxed)
              'C058', 'C059']   # Shame <-> Triumph                  (relaxed)

chosen = list(MIRROR_IDS)


def cover(ids):
    """(start region, end region) cells covered, and centroid positions."""
    cells = set()
    for i in ids:
        p = C[i]['path']
        cells.add((int(REGION[p[0]]), int(REGION[p[-1]])))
    return cells


# ------------------------------------------------------------ greedy fill ---
# maximise: new (region_start, region_end) cell, new voice family, new path
# length, new language, and geometric distance from the already-picked set.
fills = [x for x in C.values() if x['kind'] == 'fill']
cells = cover(chosen)
fams = collections.Counter(FAM(C[i]['voice']) for i in chosen)
lens = collections.Counter(len(C[i]['comps']) for i in chosen)
langs = collections.Counter(C[i]['lang'] for i in chosen)


def mid(x):
    p = x['path']
    return Z[p].mean(0)


picked_mid = [mid(C[i]) for i in chosen]

while len(chosen) < 20:
    best, bs = None, -1e9
    for x in fills:
        if x['id'] in chosen:
            continue
        p = x['path']
        cell = (int(REGION[p[0]]), int(REGION[p[-1]]))
        s = 0.0
        s += 6.0 * (cell not in cells)
        s += 3.0 / (1 + fams[FAM(x['voice'])])
        s += 2.0 / (1 + lens[len(x['comps'])])
        s += 1.5 / (1 + langs[x['lang']])
        d = min(float(np.linalg.norm(mid(x) - m)) for m in picked_mid)
        s += 0.9 * d
        # keep the page listenable: prefer 30-60 s
        t = x['audio']['total_s']
        s -= 0.05 * max(0.0, 30.0 - t) + 0.05 * max(0.0, t - 60.0)
        if s > bs:
            bs, best = s, x
    chosen.append(best['id'])
    p = best['path']
    cells.add((int(REGION[p[0]]), int(REGION[p[-1]])))
    fams[FAM(best['voice'])] += 1
    lens[len(best['comps'])] += 1
    langs[best['lang']] += 1
    picked_mid.append(mid(best))

sel = [C[i] for i in chosen]
os.makedirs(W + '/work', exist_ok=True)
json.dump([x['id'] for x in sel], open(W + '/work/selection.json', 'w'), indent=1)

print(f'{len(sel)} selected, {sum(x["audio"]["total_s"] for x in sel):.0f} s total audio')
print(f'  voice families : {dict(collections.Counter(FAM(x["voice"]) for x in sel))}')
print(f'  languages      : {dict(collections.Counter(x["lang"] for x in sel))}')
print(f'  path lengths   : {dict(sorted(collections.Counter(len(x["comps"]) for x in sel).items()))}')
print(f'  ops            : {dict(collections.Counter(x["op"] for x in sel))}')
print(f'  region cells   : {len(cells)} distinct (start,end) of 64 possible')
print(f'  start regions  : {sorted({int(REGION[x["path"][0]]) for x in sel})}')
print(f'  end regions    : {sorted({int(REGION[x["path"][-1]]) for x in sel})}')
print(f'  distinct voices: {len({x["voice"] for x in sel})}')
print()
for x in sel:
    p = x['path']
    print(f'{x["id"]}  {x["kind"]:6s} {x["op"][:4]}  {x["voice"]:18s} {x["lang"]}  '
          f'{x["audio"]["total_s"]:5.1f}s  r{int(REGION[p[0]])}->r{int(REGION[p[-1]])}  '
          + ' > '.join(c['state_name'].replace('intense ', '!').replace('moderate ', '~') for c in x['comps']))
