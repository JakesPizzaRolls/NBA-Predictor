"""Train full-feature PyTorch NBA outcome predictor with time-based validation."""
import argparse,copy,json,random
from pathlib import Path
import joblib,numpy as np,pandas as pd,torch
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score,roc_auc_score,log_loss,brier_score_loss
from torch import nn

class Net(nn.Module):
    def __init__(self,n,dropout=.25):
        super().__init__()
        self.layers=nn.Sequential(nn.Linear(n,256),nn.LayerNorm(256),nn.GELU(),nn.Dropout(dropout),nn.Linear(256,128),nn.GELU(),nn.Dropout(dropout),nn.Linear(128,64),nn.GELU(),nn.Linear(64,1))
    def forward(self,x):return self.layers(x).squeeze(-1)

def score(y,p):
    return {'accuracy':float(accuracy_score(y,p>=.5)),'roc_auc':float(roc_auc_score(y,p)) if len(set(y))>1 else None,'log_loss':float(log_loss(y,p,labels=[0,1])),'brier':float(brier_score_loss(y,p))}

def main():
    p=argparse.ArgumentParser();p.add_argument('--features',default='features.csv');p.add_argument('--out',default='nba_full_model.pt');p.add_argument('--validation-season',type=int,default=2021);p.add_argument('--test-season',type=int,default=2022);p.add_argument('--min-season',type=int,default=1946);p.add_argument('--epochs',type=int,default=80);p.add_argument('--patience',type=int,default=12);p.add_argument('--batch-size',type=int,default=256);p.add_argument('--seed',type=int,default=42);a=p.parse_args()
    random.seed(a.seed);np.random.seed(a.seed);torch.manual_seed(a.seed)
    df=pd.read_csv(a.features)
    df=df[(df.season>=a.min_season)&(df.season<=a.test_season)].copy()
    # No IDs or dates as input. They are metadata only.
    cols=[c for c in df.columns if c not in ['game_id','date','season','home_win','home_team_id','away_team_id'] and pd.api.types.is_numeric_dtype(df[c])]
    tr=df[df.season<a.validation_season];va=df[df.season==a.validation_season];te=df[df.season==a.test_season]
    if not len(tr) or not len(va) or not len(te):raise ValueError('Missing train, validation, or test seasons')
    # Entirely missing training columns are excluded rather than imputed using future data.
    cols=[c for c in cols if tr[c].notna().any()]
    imputer=SimpleImputer(strategy='median',keep_empty_features=True);scaler=StandardScaler()
    xtr=scaler.fit_transform(imputer.fit_transform(tr[cols])).astype('float32')
    xva=scaler.transform(imputer.transform(va[cols])).astype('float32')
    xte=scaler.transform(imputer.transform(te[cols])).astype('float32')
    ytr=tr.home_win.to_numpy(dtype='float32');yva=va.home_win.to_numpy(dtype='float32');yte=te.home_win.to_numpy(dtype='float32')
    model=Net(len(cols));optim=torch.optim.AdamW(model.parameters(),lr=.0007,weight_decay=.01)
    criterion=nn.BCEWithLogitsLoss()
    loader=torch.utils.data.DataLoader(torch.utils.data.TensorDataset(torch.from_numpy(xtr),torch.from_numpy(ytr)),batch_size=a.batch_size,shuffle=True)
    best=float('inf');best_state=None;best_epoch=0
    for epoch in range(a.epochs):
        model.train()
        for x,y in loader:
            optim.zero_grad();loss=criterion(model(x),y);loss.backward();nn.utils.clip_grad_norm_(model.parameters(),1.0);optim.step()
        model.eval()
        with torch.no_grad():vl=criterion(model(torch.from_numpy(xva)),torch.from_numpy(yva)).item()
        if vl<best-1e-5:best=vl;best_state=copy.deepcopy(model.state_dict());best_epoch=epoch+1
        if epoch+1-best_epoch>=a.patience:break
    model.load_state_dict(best_state);model.eval()
    with torch.no_grad():pv=torch.sigmoid(model(torch.from_numpy(xva))).numpy();pt=torch.sigmoid(model(torch.from_numpy(xte))).numpy()
    metrics={'train_games':len(tr),'validation_games':len(va),'test_games':len(te),'features':len(cols),'best_epoch':best_epoch,'validation':score(yva,pv),'test':score(yte,pt),'home_team_baseline_accuracy':float(yte.mean()),'elo_resets_each_season':True,'player_features_available':any(c.startswith('home_player') for c in cols) and tr['home_roster_players'].max()>0}
    out=Path(a.out);out.parent.mkdir(parents=True,exist_ok=True)
    torch.save({'state_dict':best_state,'input_dim':len(cols),'features':cols,'validation_season':a.validation_season,'test_season':a.test_season},out)
    joblib.dump({'imputer':imputer,'scaler':scaler,'features':cols},out.with_suffix('.preprocessing.joblib'))
    out.with_suffix('.metrics.json').write_text(json.dumps(metrics,indent=2,default=lambda x: x.item() if hasattr(x, 'item') else str(x)))
    pd.DataFrame({'game_id':te.game_id,'date':te.date,'home_win':yte.astype(int),'pred_home_win':pt}).to_csv(out.with_suffix('.predictions.csv'),index=False)
    print(json.dumps(metrics,indent=2,default=lambda x: x.item() if hasattr(x, 'item') else str(x)))
if __name__=='__main__':main()
