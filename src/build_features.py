"""Leakage-aware NBA feature engineering from the entire Wyatt Walsh basketball archive.
Only historical games update features; Elo resets to 1500 each season.
"""
import argparse, collections, csv, datetime as dt, json, re, sqlite3, zipfile
from pathlib import Path
import numpy as np
import pandas as pd

BASE_STATS=['pts','fgm','fga','fg3m','fg3a','ftm','fta','oreb','dreb','reb','ast','stl','blk','tov','pf']
WINDOWS=[5,10,20]

def season_year(x):
    s=str(x)
    # NBA season IDs are 5 digits, e.g. 22022 regular, 42022 playoffs.
    return int(s[-4:]) if len(s)>=4 else -1

def load_games(archive):
    with zipfile.ZipFile(archive) as z:
        df=pd.read_csv(z.open('csv/game.csv'),low_memory=False)
    df['date']=pd.to_datetime(df.game_date,errors='coerce')
    df['season']=df.season_id.map(season_year)
    df=df.dropna(subset=['date','game_id','team_id_home','team_id_away','pts_home','pts_away'])
    df=df[df.season.between(1946,2035)].copy()
    df['game_id']=df.game_id.astype(str).str.replace(r'\.0$','',regex=True).str.zfill(10)
    df=df.sort_values(['date','game_id']).drop_duplicates('game_id',keep='first').reset_index(drop=True)
    for c in BASE_STATS:
        for side in ['home','away']:
            k=f'{c}_{side}'
            df[k]=pd.to_numeric(df[k],errors='coerce') if k in df else np.nan
    df['home_win']=(df.pts_home>df.pts_away).astype(int)
    return df

def create_pbp_cache(archive, cache):
    """Stream entire 2GB+ PBP without loading into memory; cache player-game events.
    Does NOT assume event participation equals an official box-score appearance.
    """
    cache=Path(cache); cache.parent.mkdir(parents=True,exist_ok=True)
    if cache.exists():
        con=sqlite3.connect(cache)
        try:
            if con.execute("select val from meta where key='complete'").fetchone()==('yes',):
                print('Using existing complete PBP cache:',cache,flush=True);return
        except sqlite3.Error: pass
        finally:con.close()
        cache.unlink()
    con=sqlite3.connect(cache)
    con.execute('create table meta(key text primary key,val text)')
    con.execute('create table player_games(game_id text,team_id integer,player_id integer,events integer,points real,fgm real,fga real,fg3m real,fg3a real,ftm real,fta real,reb real,ast real,stl real,blk real,tov real,pf real,primary key(game_id,team_id,player_id))')
    sql='insert or replace into player_games values('+','.join(['?']*17)+')'
    with zipfile.ZipFile(archive) as z:
        matches=[x for x in z.namelist() if x.endswith('play_by_play.csv')]
        if not matches:raise ValueError('Full archive must contain csv/play_by_play.csv. Use --no-pbp for smaller archive.')
        with z.open(matches[0]) as fh:
            import io
            reader=csv.DictReader(io.TextIOWrapper(fh,encoding='utf-8-sig',errors='replace',newline=''))
            current=None; agg={}; n=0
            def flush():
                if agg:
                    con.executemany(sql,[(current,t,p,*v) for (t,p),v in agg.items()])
            for r in reader:
                gid=str(r.get('game_id','')).strip().zfill(10)
                if not gid or gid=='0000000000':continue
                if current is None:current=gid
                if gid!=current:
                    flush();agg={};current=gid
                    if n%500000==0:con.commit()
                try:typ=int(r.get('eventmsgtype') or 0)
                except ValueError:typ=0
                desc=' '.join([r.get('homedescription') or '',r.get('visitordescription') or '',r.get('neutraldescription') or '']).upper()
                # Players in all 3 slots are evidence of participation, not box-score totals.
                for i in (1,2,3):
                    try:p=int(r.get(f'player{i}_id') or 0);t=int(float(r.get(f'player{i}_team_id') or 0))
                    except ValueError:continue
                    if p>0 and t>0:
                        a=agg.setdefault((t,p),[0.0]*14);a[0]+=1
                try:p=int(r.get('player1_id') or 0);t=int(float(r.get('player1_team_id') or 0))
                except ValueError:p=t=0
                if p>0 and t>0:
                    a=agg[(t,p)]
                    if typ in (1,2):
                        a[3]+=1 # FGA
                        three=bool(re.search(r'3PT|3-PT|THREE POINT',desc))
                        if three:a[5]+=1
                        if typ==1:
                            a[2]+=1
                            a[1]+=3 if three else 2
                            if three:a[4]+=1
                    elif typ==3:
                        if 'MISS' not in desc and ('FREE THROW' in desc):a[1]+=1;a[6]+=1
                        if 'FREE THROW' in desc:a[7]+=1
                    elif typ==4:a[8]+=1
                    elif typ==5:a[12]+=1
                    elif typ==6:a[13]+=1
                # Assists/steals/blocks are sometimes in player2/3 slots.
                if typ==1 and 'AST' in desc:
                    try:p2=int(r.get('player2_id') or 0);t2=int(float(r.get('player2_team_id') or 0))
                    except ValueError:p2=t2=0
                    if (t2,p2) in agg:agg[(t2,p2)][9]+=1
                if typ==5 and 'STEAL' in desc:
                    try:p2=int(r.get('player2_id') or 0);t2=int(float(r.get('player2_team_id') or 0))
                    except ValueError:p2=t2=0
                    if (t2,p2) in agg:agg[(t2,p2)][10]+=1
                if typ==2 and 'BLOCK' in desc:
                    try:p2=int(r.get('player2_id') or 0);t2=int(float(r.get('player2_team_id') or 0))
                    except ValueError:p2=t2=0
                    if (t2,p2) in agg:agg[(t2,p2)][11]+=1
                n+=1
                if n%1000000==0:
                    con.commit();print('Processed PBP events:',f'{n:,}',flush=True)
            flush()
    con.commit();con.execute('create index idx_pg_game on player_games(game_id)')
    con.execute("insert into meta values('complete','yes')");con.commit();con.close()
    print('PBP cache complete:',f'{n:,}','events',flush=True)

def build(archive,out,cache=None,use_pbp=True):
    games=load_games(archive)
    if use_pbp:
        if cache is None:cache=Path(out).with_suffix('.pbp.sqlite')
        create_pbp_cache(archive,cache)
        con=sqlite3.connect(cache)
        # Game-based lookup, fast with idx_pg_game; do not prefetch entire dataset.
    else:con=None
    # Past-only player observations: (date, team, player, points, fga, ast, reb, tov, events)
    player_hist=collections.defaultdict(lambda:collections.deque(maxlen=20))
    team_player_last={}
    team_players=collections.defaultdict(set)
    history=collections.defaultdict(list)
    elo={};active_season=None
    records=[]
    feature_names=None
    for date,day in games.groupby('date',sort=True):
        # Games on same date are all predicted before updating ANY game on that date.
        for _,g in day.iterrows():
            season=int(g.season)
            if active_season!=season:
                # Season-reset rule: no carryover. Historical team rolling data can carry over,
                # but Elo begins at 1500 for every team in each new season.
                elo={};active_season=season
            home=int(g.team_id_home);away=int(g.team_id_away)
            f={'game_id':g.game_id,'date':str(date.date()),'season':season,'home_win':int(g.home_win),
               'home_team_id':home,'away_team_id':away,'elo_home':elo.get(home,1500.),'elo_away':elo.get(away,1500.),
               'elo_diff':elo.get(home,1500.)-elo.get(away,1500.),'game_type_code':int(str(g.season_id)[0]) if str(g.season_id)[0].isdigit() else 0}
            for side,team,opp in [('home',home,away),('away',away,home)]:
                hist=history[team]
                f[f'{side}_games_seen']=len(hist)
                f[f'{side}_rest_days']=min(30,(date-hist[-1][0]).days) if hist else 7
                f[f'{side}_back_to_back']=int(bool(hist) and (date-hist[-1][0]).days<=1)
                for w in WINDOWS:
                    recent=hist[-w:]
                    f[f'{side}_winrate_{w}']=np.mean([x[1] for x in recent]) if recent else .5
                    for stat in BASE_STATS+['opp_pts','possessions','off_rating','def_rating','net_rating']:
                        vals=[x[2].get(stat,np.nan) for x in recent]
                        f[f'{side}_{stat}_{w}']=float(np.nanmean(vals)) if vals and np.isfinite(vals).any() else np.nan
                # Historical roster approximation based only on prior observed events.
                roster=[]
                if con:
                    for p in team_players[team]:
                        t=team;last=team_player_last[(t,p)]
                        if (date-last).days<=45:
                            obs=[x for x in player_hist[(t,p)] if x[0]<date and (date-x[0]).days<=45]
                            if obs:roster.append((p,obs))
                f[f'{side}_roster_players']=len(roster)
                for stat,idx in [('points',1),('fga',2),('assists',3),('rebounds',4),('turnovers',5),('events',6)]:
                    vals=[np.mean([x[idx] for x in obs]) for _,obs in roster]
                    f[f'{side}_player_{stat}_mean']=float(np.mean(vals)) if vals else np.nan
                    f[f'{side}_player_{stat}_top5']=float(sum(sorted(vals,reverse=True)[:5])) if vals else np.nan
                f[f'{side}_player_usage_concentration']=float(max([sum(x[6] for x in obs) for _,obs in roster],default=0)/max(1,sum(sum(x[6] for x in obs) for _,obs in roster))) if roster else np.nan
            for stat in ['winrate_5','winrate_10','winrate_20','pts_5','opp_pts_5','net_rating_10','net_rating_20','rest_days','player_points_top5','player_assists_top5','player_rebounds_top5','player_events_top5']:
                f[f'diff_{stat}']=f.get('home_'+stat,np.nan)-f.get('away_'+stat,np.nan)
            records.append(f)
        # Apply all game results only after features for the entire date are recorded.
        for _,g in day.iterrows():
            h=int(g.team_id_home);a=int(g.team_id_away)
            eh=elo.get(h,1500.);ea=elo.get(a,1500.)
            expected=1/(1+10**((ea-eh-70)/400))
            delta=20*(int(g.home_win)-expected)
            elo[h]=eh+delta;elo[a]=ea-delta
            for side,team,other in [('home',h,'away'),('away',a,'home')]:
                own={stat:float(g[f'{stat}_{side}']) for stat in BASE_STATS}
                own['opp_pts']=float(g[f'pts_{other}'])
                poss=own['fga']-own['oreb']+own['tov']+0.44*own['fta']
                opp_poss=float(g[f'fga_{other}'])-float(g[f'oreb_{other}'])+float(g[f'tov_{other}'])+0.44*float(g[f'fta_{other}'])
                own['possessions']=poss
                own['off_rating']=100*own['pts']/poss if poss>0 else np.nan
                own['def_rating']=100*own['opp_pts']/opp_poss if opp_poss>0 else np.nan
                own['net_rating']=own['off_rating']-own['def_rating']
                history[team].append((date,int((g.home_win if side=='home' else 1-g.home_win)),own))
            if con:
                for gid,t,p,events,pts,fgm,fga,fg3m,fg3a,ftm,fta,reb,ast,stl,blk,tov,pf in con.execute('select * from player_games where game_id=?',(g.game_id,)):
                    player_hist[(t,p)].append((date,pts,fga,ast,reb,tov,events))
                    team_player_last[(t,p)]=date
                    team_players[t].add(p)
        if len(records)%10000< len(day):print('Games featured:',len(records),flush=True)
    if con:con.close()
    df=pd.DataFrame(records)
    # Keep all historical games, including playoff rows, with game-type as input.
    Path(out).parent.mkdir(parents=True,exist_ok=True)
    df.to_csv(out,index=False)
    print('Saved',len(df),'games and',len(df.columns)-7,'numeric features to',out,flush=True)
    return df

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--data',required=True);p.add_argument('--out',default='features.csv');p.add_argument('--cache',default='player_events.sqlite');p.add_argument('--no-pbp',action='store_true')
    a=p.parse_args();build(a.data,a.out,a.cache,not a.no_pbp)
