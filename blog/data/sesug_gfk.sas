option validvarname=any;

libname sesug './';

/******** State, Team, Player and All Star **********/
data aagpbl;
  set sesug.aagpbl_for_sesug;
  name = propcase(firstname||' '||lastname);
  team = propcase(team);
  if g = -99 then g = 0;
  if pct <= 0 then pct = 0;
  array num1 ab--l;
  do over num1;
    if num1 <= -99 then num1 = .;
  end;
run;

proc sql;
  create table player_home as
  select st_prov,count(distinct name) as num_player,round(sum(G)/calculated num_player,1.) as total_game,
         round(sum(g*(case when allstar = 'Y' then 1 else 0 end))/calculated num_player,1.) as game_as_allstar
  from aagpbl
  group by st_prov;
  delete from player_home where st_prov = 'UNK';
quit;

data player_home;
  set player_home;
  if num_player <= 5 then player = '1: 1 - 5           ';
  else if num_player <= 10 then player = '2: 6 - 10';
  else if num_player <= 20 then player = '3: 11 - 20';
  else if num_player <= 30 then player = '4: 21 - 30';
  else if num_player <= 50 then player = '5: 31 - 50';
  else player = '6: > 50';
  label player = 'Number of Players';
run;

data canada_mapping;
input st_prov $3. id1name $55.;
cards;
AB Alberta
BC British Columbia
MB Manitoba
NB New Brunswick
NL Newfoundland and Labrador
NT Northwest Territories
NS Nova Scotia
NU Nunavut
ON Ontario
PE Prince Edward Island
QC Quebec
SK Saskatchewan
YT Yukon
;
run;

proc sort data=mapsgfk.canada_attr nodupkey out=canada_attr(keep=id1 id1name);
  by id1;
run;

data canada_attr;
  set canada_attr;
  format province $2.;
  province = scan(id1,2,'-');
run;

data canada;
  set mapsgfk.canada;
run;
 
proc sort data=canada_attr;
  by id1;
proc sort data=canada;
  by id1;
run;

data canada;
  merge canada(in=a) canada_attr(in=b);
  by id1;
  if a and b then output;
run;

proc sort data=canada;
  by id1name;
proc sort data=canada_mapping;
  by id1name;
run;

data canada;
  merge canada(in=a) canada_mapping(in=b);
  by id1name;
  if a then output;
run;

proc sort data=canada;
  by st_prov;
run;

proc gremove data=canada out=canada_prov;
  by st_prov;
  id id1 id;
quit;

data us_all;
  set mapsgfk.us_states;
  format st_prov $3.;
  st_prov = statecode;
run;

data us_ca;
  set us_all canada_prov;
  if st_prov in ('PR','HI','AK') then delete;
  x=long;
  y=lat;
run;

proc gproject data=us_ca out=proj_us_ca project=ALBERS degrees eastlong;
  id st_prov;
quit;

data anno_pie;
  set player_home;
  format color $10. function $10. color $8.;
  size = 0.12*sqrt(total_game);
  pct_allstar = game_as_allstar/total_game;
  pct_other = 1 - pct_allstar;
  xsys='2'; ysys='2'; hsys='3'; when='A'; style='psolid';
  function = 'PIE';line=0;
  rotate = pct_other*360;
  color = "#FFD700";
  output;
  rotate = pct_allstar*360;
  color = "#ff4d4d";
  output;
run;

data anno_legend;
  format color $10. function $10. text $32. color $8.;
  xsys='3'; ysys='3'; hsys='1'; 
  when='A';style='psolid';function='PIE';line=0;
  x = 75;y = 30;size = 0.12*sqrt(400);rotate=0.8*360;color = "#FFD700";output;rotate = 0.2*360;color = "#ff4d4d";output;
  y = 23.3;size = 0.12*sqrt(350);rotate=0.8*360;color = "#FFD700";output;rotate = 0.2*360;color = "#ff4d4d";output;
  y = 17.8;size = 0.12*sqrt(250);rotate=0.8*360;color = "#FFD700";output;rotate = 0.2*360;color = "#ff4d4d";output;
  y = 12.9;size = 0.12*sqrt(200);rotate=0.8*360;color = "#FFD700";output;rotate = 0.2*360;color = "#ff4d4d";output;
  y = 9;size = 0.12*sqrt(150);rotate=0.8*360;color = "#FFD700";output;rotate = 0.2*360;color = "#ff4d4d";output;
  y = 5.7;size = 0.12*sqrt(100);rotate=0.8*360;color = "#FFD700";output;rotate = 0.2*360;color = "#ff4d4d";output;
  y = 2.8;size = 0.12*sqrt(50);rotate=0.8*360;color = "#FFD700";output;rotate = 0.2*360;color = "#ff4d4d";output;
  function='label'; position='5'; size=2; color = "black"; style='times amt';rotate = 360;
  x = 85;y = 30;text="400+ Games Per Player/%All Star"; output;
  y = 23.5;text="350+ Games Per Player/%All Star"; output;
  y = 18.1; text="250+ Games Per Player/%All Star"; output;
  y = 13.2; text="200+ Games Per Player/%All Star"; output;
  y = 9.6; text="150+ Games Per Player/%All Star"; output;
  y = 6.2; text="100+ Games Per Player/%All Star"; output;
  y = 3; text="50+ Games Per Player/%All Star"; output;
run;

proc sort data=proj_us_ca;
  by st_prov;
run;

%ANNOMAC;
%centroid(proj_us_ca,center_us_ca,st_prov);

proc sql;
  create table anno_pie as
  select a.*,b.x,b.y
  from anno_pie a left join center_us_ca b
  on a.st_prov = b.st_prov;
quit;

data all_anno;
  set anno_pie anno_legend;
run;

goptions reset=all border cback=white htitle=12pt htext=10pt xpixels=1200 ypixels=800;
legend1 label=none shape=bar(3,1) value=(justify=left '1 - 5 Players' '6 - 10 Players' '11 - 20 Players' '21 - 30 Players' '31 - 50 Players' '51 - 97 Players') 
        across=1 origin=(5,2) mode=share;
pattern1 value=msolid color='#c7e9b4';
pattern2 value=msolid color='#7fcdbb';
pattern3 value=msolid color='#41b6c4';
pattern4 value=msolid color='#1d91c0';
pattern5 value=msolid color='#225ea8';
pattern6 value=msolid color='#0c2c84';
title1 "Teams/Players/Allstars from State/Province";
proc gmap data=player_home map=proj_us_ca all density=2 anno=all_anno;
  id st_prov;
  choro player / discrete coutline=black cdefault=gwh legend=legend1;
run;
quit;
title1;
/******** End State, Team, Player and All Star **********/

/******** Team Win Losses by Year **********/
proc sql;
  create table team_win_loss as 
  select year,team,mean(team_w) as team_w,mean(team_l) as team_l,
         int(100*calculated team_w/(calculated team_w+calculated team_l)) as pct_w,
		 sum(case when allstar = 'Y' then 1 else 0 end)/count(distinct name) as pct_allstar
  from aagpbl
  group by year,team
  order by year,team;
quit;

ODS graphics on / reset width=1900 height=1200 scale=on;
title1 'Team Wins/Losses Percentage with All Stars Percentage from Year 1943 - 1954';
proc sgplot data=team_win_loss;                                                                                                                
   bubble x=year y=pct_w size=pct_allstar/ group=team bradiusmin=4 bradiusmax=12 transparency=0.3; 
   xaxis values=(1943 to 1954 by 1);
   yaxis values=(0 to 100 by 10) label='Percentage of Team Win';
   inset "Bubble size represents Percentage of All Star Players" / position=bottomright textattrs=(size=11);
run;                                                                                                                                    
quit;
/******** End Team Win Losses by Year **********/

/******** Player Performance  *************/

data aagpbl_radar;
  set aagpbl;
  if position = 'UNK' then delete;
  avg_AB = ab/g;
  avg_R = r/g;
  avg_H = h/g;
  avg_2B = '2b'n/g;
  avg_3B = '3b'n/g;
  avg_HR = hr/g;
  avg_RBI = rbi/g;
  avg_SB = sb/g;
  avg_BB = bb/g;
  avg_SO = so/g;
  mean_IP = ip/pg;
  mean_RA = ra/pg;
  mean_ER = er/pg;
  mean_ERA = era/pg;
  mean_PBB = pbb/pg;
  mean_PSO = pso/pg;
  mean_HB = hb/pg;
  mean_WP = wp/pg;
run;

proc sort data=aagpbl_radar;
  by position;
run;

proc transpose data=aagpbl_radar out=aagpbl_tr name=play_type;
  by position;
  var avg: mean_:;
run;

data aagpbl_tr;
  set aagpbl_tr;
  num_success = sum(of col1-col498);
  count = 0;
  array non_miss{498} col1-col498;
  do i = 1 to 498;
    if non_miss{i} ^=. then count = count+1;
  end;
  drop col: i;
  avg_success = num_success/count*100;
  if lowcase(scan(play_type,1,'_')) = 'avg' then type = 1;
  else type = 2;
  if avg_success <= 0 then avg_success = 0.01;
  if play_type = 'AVG' then success_type = 'Batting                         ';
  else if play_type = 'avg_2B' then success_type = 'Doubles';
  else if play_type = 'avg_3B' then success_type = 'Triples';
  else if play_type = 'avg_AB' then success_type = 'At Bats';
  else if play_type = 'avg_BB' then success_type = 'Walks (Bases on Balls)';
  else if play_type = 'avg_H' then success_type = 'Hits';
  else if play_type = 'avg_HR' then success_type = 'Home Runs';
  else if play_type = 'avg_R' then success_type = 'Runs';
  else if play_type = 'avg_RBI' then success_type = 'Runs Batted In';
  else if play_type = 'avg_SB' then success_type = 'Stolen Bases';
  else if play_type = 'avg_SO' then success_type = 'Strike Outs by Batter';
  else if play_type = 'mean_ER' then success_type = 'Earned Runs';
  else if play_type = 'mean_ERA' then success_type = 'Earned Run';
  else if play_type = 'mean_HB' then success_type = 'Hit by Pitch';
  else if play_type = 'mean_IP' then success_type = 'Innings Pitched';
  else if play_type = 'mean_PBB' then success_type = 'Walks (Bases on Balls) Allowed';
  else if play_type = 'mean_PSO' then success_type = 'Strike Outs by Pitcher';
  else if play_type = 'mean_RA' then success_type = 'Runs Allowed';
  else if play_type = 'mean_WP' then success_type = 'Wild Pitches';
run;

proc sort data=aagpbl_tr;
  by play_type;
run;

proc stdize data=aagpbl_tr out=aagpbl_tr_sd method=range;
  by play_type;
  var avg_success;
run;

data aagpbl_tr_sd;
  set aagpbl_tr_sd;
  avg_success = 100*avg_success;
  if avg_success = 0 then avg_success = 0.0001;
run;

goptions reset=all border cback=white htitle=12pt htext=10pt xpixels=1200 ypixels=800;
title1 'Standarized Average Actions by Positions';
proc gradar data=aagpbl_tr_sd;
    where type = 1;
    chart success_type / overlayvar=position freq=avg_success
                         cstars=(red, lilac, magenta, gold, tan, lime, steel, charcoal, purple, green, cyan)
                         wstars=2 2 2 2 2 2 2 2 2 2 2
                         lstars=1 1 1 1 1 1 1 1 1 1 1
                         starcircles=(0.5 1.0) cstarcircles=ltgray;
run;
quit;

goptions reset=all border cback=white htitle=12pt htext=10pt xpixels=1200 ypixels=800;
title1 "Standarized Average Actions by Positions When Becoming Pitchers";
proc gradar data=aagpbl_tr_sd;
    where type = 2;
    chart success_type / overlayvar=position freq=avg_success 
                         cstars=(red, lilac, magenta, gold, tan, lime, steel, charcoal) 
                         wstars=2 2 2 2 2 2 2 2 
                         lstars=1 1 1 1 1 1 1 1
                         starcircles=(0.5 1.0) cstarcircles=ltgray;
run;
quit;