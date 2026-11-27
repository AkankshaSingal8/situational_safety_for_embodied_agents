import numpy as np,glob,os,sys
for root in sys.argv[1:]:
    for t in sorted(os.listdir(root)):
        n=ns=c=cs=0
        for f in glob.glob(f"{root}/{t}/seed*.npz"):
            z=np.load(f,allow_pickle=True); hf=z["health_final"]; safe=bool(z["success_any"]) and (hf.size==0 or hf.min()>=100-1e-6)
            res=z["exec_xyz"]-z["base_xyz"]; k=int((z["flag"].astype(bool)&(np.abs(res).max(axis=(1,2))>1e-6)).sum())
            n+=1;c+=k
            if safe: ns+=1;cs+=k
        print(root.split("/")[-1],t,"eps",n,"corr_all",c,"| safe_eps",ns,"corr_safe",cs,flush=True)
