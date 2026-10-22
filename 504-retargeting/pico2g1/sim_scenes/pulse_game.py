"""Read-only task scoring. Call after mj_step; no robot actuator commands are issued."""
import numpy as np
class ArenaGame:
 def __init__(self,model,design):
  self.model=model;self.design=design;self.reset()
 def reset(self):
  self.score=0;self.completed=set();self.button_index=0;self.waypoint_index=0;self.pressed={};self.since={};self.last_time=None
 def snapshot(self):
  return {'score':self.score,'max_score':135,'button_index':self.button_index,'waypoint_index':self.waypoint_index,'completed':sorted(self.completed)}
 def update(self,data,actor_xy=None):
  now=float(data.time)
  if self.last_time is not None and now<self.last_time:self.reset()
  self.last_time=now;events=[]
  def award(key,points):
   if key not in self.completed:
    self.completed.add(key);self.score+=points;events.append({'task':key,'points':points,'time':now,'score':self.score})
  def dwell(key,condition,seconds):
   if not condition:self.since.pop(key,None);return False
   self.since.setdefault(key,now);return now-self.since[key]>=seconds
  for i,b in enumerate(self.design['buttons']):
   name=b['name'];travel=float(data.joint('button_'+name+'_slide').qpos[0]);old=self.pressed.get(name,False)
   pressed=travel>.006 if old else travel>=.018
   if pressed and not old and i==self.button_index:award('button_'+name,10);self.button_index+=1
   self.pressed[name]=pressed
  for item in self.design['cubes']:
   n='cube_'+item['name'];body=data.body(n);target=np.array(item['target']);ext=np.abs(body.xmat.reshape(3,3))@np.full(3,self.design['cube_half_size'])
   contained=np.all(np.abs(body.xpos[:2]-target[:2])+ext[:2]<.295) and .81<body.xpos[2]-ext[2] and body.xpos[2]+ext[2]<1.05
   stopped=np.linalg.norm(data.joint(n+'_free').qvel)<.08
   # No contact with another moving body or mocap gripper: cube must be released.
   held=False;gid=self.model.geom(n+'_collision').id
   for contact in data.contact:
    if gid in (contact.geom1,contact.geom2):
     other=contact.geom2 if contact.geom1==gid else contact.geom1;bid=self.model.geom_bodyid[other];weld=self.model.body_weldid[bid]
     held |= self.model.body_dofnum[weld]>0 or self.model.body_mocapid[weld]>=0
   if dwell(n,bool(contained and stopped and not held),.5):award(n,20)
  ball=data.body('game_ball').xpos;goal=np.array(self.design['goal']['center']);half=np.array(self.design['goal']['half_size']);radius=self.design['ball']['radius']
  inside=np.all(np.abs(ball[:2]-goal[:2])+radius<half[:2]) and radius-.02<ball[2]<.42
  if dwell('goal',bool(inside),.3):award('goal',20)
  if actor_xy is not None and self.waypoint_index<len(self.design['waypoints']):
   target=self.design['waypoints'][self.waypoint_index]
   if np.linalg.norm(np.array(actor_xy)[:2]-target)<.4:
    award('waypoint_'+str(self.waypoint_index+1),5);self.waypoint_index+=1
  return events
