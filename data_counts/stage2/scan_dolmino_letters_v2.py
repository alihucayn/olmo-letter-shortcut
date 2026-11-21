#!/usr/bin/env python
"""v2 scan: U_strict (recount), U_sameline, U_newline, U_marker, U_token for X in A-D and 1-4,
plus U_newline_Qdoc = U_newline matches inside documents that contain a line exactly '(Q).'.
Text scanned = the `text` field of each JSONL record, each document separately; re.MULTILINE.
usage: --list work.jsonl --out out.jsonl --dl DIR --procs N"""
import urllib.error, argparse, io, gzip, json, os, re, sys, time, random, hashlib, urllib.request
from multiprocessing import Pool
try:
    import orjson; jloads = orjson.loads
except Exception:
    jloads = json.loads
try:
    import zstandard
except Exception:
    zstandard = None

SETS = {'letters': 'A-D', 'digits': '1-4'}
CUE = r'\b(?:[Aa]nswer|correct (?:option|choice))(?: is)?:?'
def pats(cls):
    g = '([%s])' % cls
    return {
     'U_strict':   CUE + r'\s?\(?%s\)?\.?[ \t]*$' % g,
     'U_sameline': CUE + r'[ \t]?\(?%s\)?\.?[ \t]*$' % g,
     'U_newline':  CUE + r'\n\(?%s\)?\.?[ \t]*$' % g,
     'U_marker':   r'[Aa]nswer:\n\(%s\)\.[ \t]*\n(?=\S)' % g,
     'U_token':    r'\bAnswer: %s[ \t]*$' % g,
    }
def patsT(cls):
    g = '([%s])' % cls
    return {  # literal newline terminator, no \\b, no MULTILINE needed (mirrors stage-1 infini-gram strings)
     'U_T14': r'(?:Answer: %s|answer is %s\.?|Answer: \(%s\))\n' % (g, g, g),
     'T1': r'Answer: %s\n' % g, 'T2': r'answer is %s\n' % g, 'T3': r'answer is %s\.\n' % g, 'T4': r'Answer: \(%s\)\n' % g,
    }
RXG = {'main': {s: {k: re.compile(v, re.M) for k, v in pats(c).items()} for s, c in SETS.items()},
       'T': {s: {k: re.compile(v) for k, v in patsT(c).items()} for s, c in SETS.items()}}
# U_T14 has three capture groups (one per alternative): label = first non-None group
GATEG = {'main': None, 'T': None}
GATE = {'U_strict': ('nswer', 'correct'), 'U_sameline': ('nswer', 'correct'), 'U_newline': ('nswer', 'correct'),
        'U_marker': ('nswer',), 'U_token': ('Answer: ',), 'U_T14': ('nswer',), 'T1': ('Answer: ',), 'T2': ('answer is',), 'T3': ('answer is',), 'T4': ('Answer: (',)}
QLINE = re.compile(r'^\(Q\)\.$', re.M)
SAMPLEDG = {'main': ('U_sameline', 'U_newline', 'U_token'), 'T': ('U_T14',)}
RES = 50

def lab(m): return next(g for g in m.groups() if g is not None)

def ctx(text, m):
    s, e = m.span()
    return {'before': text[max(0, s - 200): s], 'match': text[s:e], 'after': text[e: e + 100]}

def open_stream(fp):
    raw = open(fp, 'rb')
    if fp.endswith('.gz'): return gzip.GzipFile(fileobj=raw, mode='rb')
    if fp.endswith('.zst'):
        return io.BufferedReader(zstandard.ZstdDecompressor().stream_reader(raw), buffer_size=1 << 20)
    return raw

def download(path, dst, tries=12):
    url = 'https://huggingface.co/datasets/allenai/dolmino-mix-1124/resolve/main/' + path
    for t in range(tries):
        wait = 5 * (t + 1)
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers={'User-Agent': 'dataset-scan'}), timeout=120) as r, open(dst, 'wb') as f:
                while True:
                    b = r.read(8 << 20)
                    if not b: break
                    f.write(b)
            return t
        except urllib.error.HTTPError as e:
            err = repr(e)
            if e.code == 429:
                try: wait = max(wait, int(e.headers.get('Retry-After', 0)) + 2)
                except Exception: wait = 60
                wait = max(wait, 30 * (t + 1))
            time.sleep(wait)
        except Exception as e:
            err = repr(e); time.sleep(wait)
    raise RuntimeError('download failed ' + path + ' ' + err)

def scan(job):
    subset, path, size, dldir, grp = job['subset'], job['path'], job['size'], job['dl'], job['group']
    RX, SAMPLED = RXG[grp], SAMPLEDG[grp]
    t0 = time.time()
    fp = os.path.join(dldir, hashlib.md5(path.encode()).hexdigest() + '_' + os.path.basename(path))
    try:
        retries = download(path, fp)
        dl_s = time.time() - t0
        rng = random.Random('0|' + path)
        cnt = {}; docsc = {}; res = {}; seen = {}
        for s in SETS:
            for p in SAMPLED: res[(p, s)] = []; seen[(p, s)] = 0
        ndocs = nbytes = nchars = nfail = nnotext = nQdocs = 0
        with open_stream(fp) as fh:
            for line in fh:
                if not line.strip(): continue
                try: rec = jloads(line)
                except Exception: nfail += 1; continue
                text = rec.get('text') if isinstance(rec, dict) else None
                if not isinstance(text, str): nnotext += 1; continue
                ndocs += 1; nchars += len(text); nbytes += len(text.encode('utf-8', 'replace'))
                hasq = None
                for s, rx in RX.items():
                    local = {}
                    for p, r in rx.items():
                        if not any(w in text for w in GATE[p]): continue
                        ms = list(r.finditer(text))
                        if not ms: continue
                        for m in ms: local[(p, lab(m))] = local.get((p, lab(m)), 0) + 1
                        if grp == 'main' and p == 'U_newline':
                            if hasq is None:
                                hasq = QLINE.search(text) is not None
                                if hasq: nQdocs += 1
                            if hasq:
                                for m in ms: local[('U_newline_Qdoc', lab(m))] = local.get(('U_newline_Qdoc', lab(m)), 0) + 1
                        if p in SAMPLED:
                            R = res[(p, s)]
                            for m in ms:
                                seen[(p, s)] += 1; n = seen[(p, s)]
                                e = {'label': lab(m), 'doc_id': rec.get('id'), **ctx(text, m)}
                                if len(R) < RES: R.append(e)
                                else:
                                    j = rng.randrange(n)
                                    if j < RES: R[j] = e
                    for k, v in local.items():
                        key = '%s|%s|%s' % (k[0], s, k[1]); cnt[key] = cnt.get(key, 0) + v; docsc[key] = docsc.get(key, 0) + 1
        os.remove(fp)
        return {'subset': subset, 'shard': path, 'file_bytes': size, 'docs_scanned': ndocs, 'text_bytes': nbytes, 'text_chars': nchars,
                'group': grp, 'parse_fail': nfail, 'no_text_field': nnotext, 'docs_with_Q_line_and_newline_match': nQdocs, 'dl_retries': retries, 'dl_s': round(dl_s, 1), 'total_s': round(time.time() - t0, 1),
                'counts': {k: [v, docsc[k]] for k, v in cnt.items()},
                'samples': {'%s|%s' % k: {'n': seen[k], 'res': res[k]} for k in res}}
    except Exception as e:
        if os.path.exists(fp): os.remove(fp)
        return {'subset': subset, 'shard': path, 'error': repr(e)[:500]}

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--list'); ap.add_argument('--out'); ap.add_argument('--dl'); ap.add_argument('--procs', type=int, default=4); ap.add_argument('--group', default='main', choices=['main','T'])
    a = ap.parse_args()
    os.makedirs(a.dl, exist_ok=True)
    jobs = [json.loads(l) for l in open(a.list)]
    done = set()
    if os.path.exists(a.out):
        for l in open(a.out):
            try:
                r = json.loads(l)
                if 'error' not in r: done.add(r['shard'])
            except Exception: pass
    if os.path.exists(a.out) and os.path.getsize(a.out) > 0:
        with open(a.out, 'rb+') as fb:
            fb.seek(-1, 2)
            if fb.read(1) != b'\n': fb.write(b'\n')
    jobs = [dict(j, dl=a.dl, group=a.group) for j in jobs if j['path'] not in done]
    print('todo', len(jobs), 'done', len(done), flush=True)
    t0 = time.time()
    with Pool(a.procs) as pool, open(a.out, 'a') as fo:
        for i, r in enumerate(pool.imap_unordered(scan, jobs)):
            fo.write(json.dumps(r) + '\n'); fo.flush()
            print(i + 1, len(jobs), r.get('shard'), r.get('total_s'), r.get('error', ''), round(time.time() - t0), flush=True)
