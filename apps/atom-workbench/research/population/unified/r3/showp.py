import sys,json
src=open(sys.argv[1]) if len(sys.argv)>1 else sys.stdin
last=int(sys.argv[2]) if len(sys.argv)>2 else 0
rows=[]
for line in src:
    line=line[line.find('{'):] if '{' in line else ''
    if not line: continue
    try: r=json.loads(line)
    except: print(line[:300]); continue
    if r.get('probe')=='pipeline': rows.append(r)
for r in rows[-last:] if last else rows:
    if 'error' in r: print(r['case'],r['run'],'ERROR',r['error'][:300]); continue
    f=r.get('finite') or {}
    sa=r.get('steadyAttempts') or []
    st=' '.join(f"o{a['order']}:{a['steps']}+{a['augmentation']}" for a in sa)
    tm=f.get('timing') or {}
    tline=' '.join(f"{k}={v:.0f}" for k,v in tm.items() if isinstance(v,(int,float)))
    d=r['difference']; ds='-' if d is None else '%.1e'%d
    est=f.get('estimate'); es='-' if est is None else '%.2e'%est
    pc=f.get('perColumn'); pcs='-' if pc is None else '%.2f'%pc
    print(f"{r['case']:22s} {r['run']:14s} ms {r['ms']:7.0f} diff {ds} | cols {f.get('columns')} inner {f.get('innerSteps')} perCol {pcs} apps {f.get('applications')} order {f.get('order')} est {es} | steady {st} asym {r.get('asymptoticError')}")
    if tline: print('      '+tline)
