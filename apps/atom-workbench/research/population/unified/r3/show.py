import sys,json
for line in sys.stdin:
    line=line[line.find('{'):] if '{' in line else line
    try: r=json.loads(line)
    except: 
        if line.strip(): print(line.strip()[:300])
        continue
    s=r.get('summary',{})
    if r.get('probe')=='solve':
        print(f"{r['case']:22s} s={r['sigma']:<6g} {r['method']:24s} perCol {s['perColumn']:6.2f} steps {s['steps']:5d} pre {r['preWork']/1e6:6.2f}M op {sum(r['opWork'])/1e6:6.2f}M | ms tot {s['totalMs']:7.0f} app {s['applyMs']:6.0f} pre {s['preMs']:6.0f} orth {s['orthMs']:6.0f} upd {s['updateMs']:5.0f} res {s['worstResidual']:.1e} K {len(r.get('support') or [0])} {[round(k,3) for k in (r.get('kernels') or [])]}")
    else: print(json.dumps(r)[:600])
