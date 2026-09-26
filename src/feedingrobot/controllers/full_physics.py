"""Independent kinematic/dynamic reconstruction; no mj_step or mutable global model."""
import numpy as np
import mujoco

from feedingrobot.controllers.so3 import orientation_error
from feedingrobot.sim.scene import FeedingScene


def check_physics(data, case):
    scene = FeedingScene('configs/m1_scene.json')
    m,s,ix = scene.model,scene.data,scene.index
    cols=ix.arm_dof_adr
    jp,jr=np.zeros((3,m.nv)),np.zeros((3,m.nv))
    full=np.empty((m.nv,m.nv))
    maxima={key:0. for key in ('tcp_position','tcp_rotation','tcp_velocity','bias','task','null','raw','wrench_transform')}
    n=len(data['tau_cmd'])
    physical=data['physical_dq']
    observed=data['dq']
    expected=physical.copy()
    if case.family=='stop' and case.event=='speed':
        dt=data['t'][1]-data['t'][0]
        expected[round(1.42/dt)+1:round(1.422/dt)+1,0]=.51
    reasons=[]
    if not np.allclose(observed,expected,atol=1e-12,rtol=0):reasons.append({'code':'observed_velocity'})
    if np.max(np.abs(physical))>.5+1e-9:reasons.append({'code':'physical_joint_speed'})
    q0=data['q'][0]
    scale=np.diag([1.,1.,1.,.1,.1,.1])
    for k in range(n+1):
        s.qpos[ix.arm_qpos_adr]=data['q'][k]
        s.qvel[cols]=physical[k]
        mujoco.mj_forward(m,s)
        mujoco.mj_jacSite(m,s,jp,jr,ix.site_ids['tcp'])
        jac=np.vstack([jp[:,cols],jr[:,cols]])
        p=s.site_xpos[ix.site_ids['tcp']]
        r=s.site_xmat[ix.site_ids['tcp']].reshape(3,3)
        for key,error in (('tcp_position',p-data['tcp_pos'][k]),('tcp_rotation',r-data['tcp_mat'][k]),('tcp_velocity',jac@physical[k]-data['tcp_twist'][k])):
            maxima[key]=max(maxima[key],float(np.max(np.abs(error))))
        if k and np.all(np.isfinite(data['raw_wrench_sensor'][k])):
            from feedingrobot.sim.sensors import world_and_tcp_wrench
            ft=ix.site_ids['ft_site']
            _,wrench=world_and_tcp_wrench(data['raw_wrench_sensor'][k],s.site_xmat[ft].reshape(3,3),s.site_xpos[ft],p,scene.wrench_sign)
            maxima['wrench_transform']=max(maxima['wrench_transform'],float(np.max(np.abs(wrench-data['ft_tcp_wrench_world'][k-1]))))
        if k==n:continue
        bias=s.qfrc_bias[cols]
        maxima['bias']=max(maxima['bias'],float(np.max(np.abs(bias-data['tau_bias'][k]))))
        if data['execution'][k]=='stop':continue
        err=np.r_[data['p_ref'][k]-p,orientation_error(data['r_ref'][k],r)]
        task=jac.T@(data['k'][k]*err+data['d'][k]*(data['v_ref'][k]-jac@observed[k]))
        mujoco.mj_fullM(m,s,full)
        mass=full[np.ix_(cols,cols)]
        js=scale@jac
        singular=np.linalg.svd(js,compute_uv=False)
        rho=singular[-1]/singular[0]
        fraction=np.clip((rho-.01)/.04,0,1)
        epsilon=np.exp(fraction*np.log(1e-8)+(1-fraction)*np.log(1e-2))
        inverse=np.linalg.solve(mass,js.T)
        a=js@inverse
        regularizer=epsilon*max(float(np.trace(a)/6),1e-4)
        jbar=inverse@np.linalg.inv(a+regularizer*np.eye(6))
        null=(np.eye(7)-jbar@js).T@(fraction*np.clip(2*(q0-data['q'][k])-observed[k],-2,2))
        raw=bias+task+null
        for key,value in (('task',task),('null',null),('raw',raw)):
            maxima[key]=max(maxima[key],float(np.max(np.abs(value-data['tau_'+key][k]))))
    for key,error in maxima.items():
        if error>1e-7:reasons.append({'code':'physics:'+key,'observed':error})
    return reasons


def check_sensor_chain(data,case,seed,dt):
    from feedingrobot.controllers.wrench import tool_body_ids
    scene=FeedingScene('configs/m1_scene.json')
    bodies=tool_body_ids(scene.model,scene.index.tool_body_id)
    masses=scene.model.body_mass[bodies]
    v=data['ft_tool_velocity']
    acceleration=np.diff(np.concatenate([data['initial_tool_velocity'][None],v]),axis=0)/dt
    forces=masses[None,:,None]*(scene.model.opt.gravity-acceleration[:,:,3:])
    omega=v[:,:,:3]
    inertia=data['ft_tool_inertia']
    moments=(np.cross(data['ft_tool_com']-data['ft_tcp_position'][:,None,:],forces)
             -np.einsum('nbij,nbj->nbi',inertia,acceleration[:,:,:3])
             -np.cross(omega,np.einsum('nbij,nbj->nbi',inertia,omega)))
    predicted=np.concatenate([forces.sum(axis=1),moments.sum(axis=1)],axis=1)
    estimate=data['ft_tcp_wrench_world']-data['ft_bias_wrench_tcp']-predicted
    valid=np.asarray(data['ft_estimated_valid'],dtype=bool)
    reasons=[]
    if not np.allclose(data['ft_estimated_kinematics'][valid],estimate[valid],atol=1e-8,rtol=1e-7):reasons.append({'code':'estimated_chain'})
    compensated=data['ft_tcp_wrench_world']-data['ft_bias_wrench_tcp']-data['ft_tool_load_predicted']
    finite=np.all(np.isfinite(compensated),axis=1)
    if not np.allclose(compensated[finite],data['ft_compensated_wrench_tcp'][finite],atol=1e-9,rtol=1e-7):reasons.append({'code':'compensation_chain'})
    filtered=data['initial_compensated'].copy()
    observed=[filtered.copy()]
    rng=np.random.default_rng(seed*10+(int(case.event[-1]) if case.event.startswith('sensor') else 0))
    alpha=1-np.exp(-2*np.pi*30*dt)
    filter_elapsed=0.
    for k,sample in enumerate(compensated):
        if not finite[k]:continue
        filtered+=alpha*(sample-filtered)
        filter_elapsed+=dt
        if not np.allclose(filtered,data['ft_filtered_wrench_tcp'][k],atol=1e-9,rtol=1e-7):
            reasons.append({'code':'filter_chain','tick':k+1});break
        measurement=filtered.copy()
        if case.noise and filter_elapsed>=1/30:
            measurement[:3]+=rng.normal(0,.02,3)
            measurement[3:]+=rng.normal(0,.002,3)
        observed.append(measurement)
        delivery=k+1-round(case.delay/dt)
        if delivery>=0 and not np.allclose(data['ft_delivered_wrench_tcp'][k],observed[delivery],atol=1e-9,rtol=1e-7):
            reasons.append({'code':'delivery_chain','tick':k+1});break
    return reasons
