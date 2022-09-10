# %%
import pandas as pd
import numpy as np
import logging
from pandas.api.types import is_numeric_dtype
# %%
### Beräkna bins av vodds
def bins(df):
    import numpy as np
    import pandas as pd

    bins = pd.DataFrame({
        'fr': [1.0, 2.1,  2.9,   3.6,  4.1,  4.8,
               5.6, 6.3,  7.1,   7.9,  8.7,  9.7,
               10.7, 11.9, 13.2, 14.8, 16.5, 18.1,
               20.2, 22.5, 25.2, 28.0, 31.5, 35.3,
               39.6, 44.7, 50.0, 56.7, 64.3, 75.4],

        'to': [2.1, 2.9,   3.6,  4.1,  4.8, 5.6,
               6.3, 7.1,   7.9,  8.7,  9.7, 10.7,
               11.9, 13.2, 14.8, 16.5, 18.1, 20.2,
               22.5, 25.2, 28.0, 31.5, 35.3, 39.6,
               44.7, 50.0, 56.7, 64.3, 75.4, 2000],

        'bin': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9,
                10, 11, 12, 13, 14, 15, 16, 17, 18, 19,
                20, 21, 22, 23, 24, 25, 26, 27, 28, 29]
    })

    
    # Alternativ lösning
    res=np.dot((df['vodds'].values[:,None] >= bins['fr'].values) &
              (df['vodds'].values[:,None] < bins['to'].values),
              bins['bin']
    )
    df['bins'] = res

    return df

# %%
#### gräv ut 'g' och 'a' ur kmtd ###
def justkm(kmtid):
    km=kmtid
    if type(kmtid)== float :
        return kmtid,False,False
    try:
        status=0
        kmtid=str.replace(kmtid,' ','')
        if kmtid is np.nan or kmtid is None or kmtid=='':
            # print('NaN')
            return np.nan,False,False
        status = 1
        galopp = False
        if km[-1] == 'g':
            km = km[0:-1]
            galopp = True
        status = 2
        auto = False
        if (len(km) > 0):
            if (km[-1] == 'a'):
                km = km[0:-1]
                auto = True
      
        status = 3
        strtid = str.split(km,',')
        status= 4
        #print(float(kmtid))
        try:
            status = 5
            tiondelar = float(strtid[1][0])
            status = 6
            sek = float(strtid[0])
            status = 7
            return sek+tiondelar/10, auto, galopp
        except:
            status=8
            return 30, auto,galopp

    except:
        print('vafalls status=',status,'km =',km,'kmtid =',kmtid)
# %%
def fixa_mer_features(df, hist=False):
    print('startar Fixa mer, ', end='')
    # get avd in the last row
    avd = df.iloc[-1]['avd']
    print('avd=', avd)
    
    ### Strukna ###
    före=len(df)
    try:
        df=df.loc[df.plac!='s'].copy()  # ta bort alla strukna - nu försvinner nog alla vodds=EJ
        print('tog först bort strukna från', före, 'till', len(df))
    except:
        pass # plac saknas    
    
    
    df=df.loc[df.vodds!='EJ',:].copy()  ## Ta bort all strukna
    efter=len(df)
    if före-efter !=0:
        print('tog nu bort',före-efter,'strukna från',före,'till',efter)

    ### Dubletter ###
    före=len(df)
    df.drop_duplicates(['datum','avd', 'häst'], inplace=True)
    efter=len(df)
    if före-efter != 0:
        print('tog bort',före-efter,'dubletter. Från',före,'till',efter)
    
    df.sort_values(by=['datum', 'bana', 'avd'], inplace=True)
    df.reset_index(drop=True, inplace=True)
      
    ### plac ###
    # diskad pga av galopp eller annat     = 20 (förr 14)
    # fullföljer men utanför prislistan (0) = 15 (förr 13)
    try:
        df['plac'] = df.plac.str.strip()
        df.loc[df.plac == 'utg', 'plac'] = 20  ## utgått pga galopp
        df.loc[df.plac=='u',['plac']]=20       ## utgått av någon anledning
        df.loc[df.plac=='d',['plac']]=20       ## diskad av någon anledning
        df.loc[df.plac=='0',['plac']]=15       ## kom utanför prislistan
        df.loc[df.plac==0,['plac']]=15         ## kom utanför prislistan
        df.loc[df.plac == '0:a', 'plac'] = 15  ## kom utanför prislistan
        df.loc[df.plac == '1:a', 'plac'] = 1
        
        df['plac'] = df.loc[:, 'plac'].astype('int')
    except:
        print('plac saknas eller är felaktig')

    ### gör start numeric ###
    df.start = (df.start == 'VOLTSTART')*1
    df['start'] = df.start.astype('int64')
    df.start.value_counts()

    ### gör vodds numeric ###
    try:
        df['vodds'] = df.vodds.str.replace(',', '.').astype('float')
    except:
        df['vodds'] = df.vodds.astype('float')

    ### gör podds numeric ###
    if df.podds.dtype==float or df.podds.dtype==int:
        pass
    else:
        # räkna om 'x.xx - y.yy' till ett medelvärde
        def new_podds(p):
            try:
                sp = p.split('-')
                x = sp[0]
                if len(sp) == 2:
                    x = (float(sp[0]) + float(sp[1])) / 2
                    return x
                else:
                    return p
            except:
                # print('podds utan intervall')
                return p

        df.podds = df.podds.str.replace(',', '.')
        if df.podds.dtype == 'O':
            df['podds'] = df['podds'].apply(lambda x: new_podds(x))

        df.loc[df.podds.isna(),'podds'] = df.vodds[df.podds.isna()]/6
        df['podds'] = df.podds.astype('float')

        ### df.spår numeriskt ###
        df.loc[df.spår == '','spår'] = np.nan
        df.loc[df.spår == ' ','spår'] = np.nan
        df.spår = df.spår.astype(float)
        df.dist = df.dist.astype(float)
    
    # bana - kolla antal NaN
    if df.bana.isna().sum()>0:
        print('bana NaNs', df.bana.isna().sum())

    # kusk
    if df.kusk.isna().sum() >0:
        print('kusk NaNs', df.kusk.isna().sum())
    
    # avd
    if df.avd.isna().sum() >0:
        print('avd NaNs', df.avd.isna().sum())
    
    # spår  NaN
    if len(df[df.spår == 0]):
        print('spår==0', df[df.spår == 0][['datum', 'bana', 'avd', 'häst', 'spår', 'start', 'vodds']])
    if df.spår.isna().sum() >0:
        print('spår NaNs', df.spår.isna().sum())
    
    # kr - ta bort blank-avskiljare i nummer
    if not is_numeric_dtype(df.kr):
        df['kr'] = df.kr.str.replace(' ', '').astype('float')

    # Ålder
    if df.ålder.isna().sum() >0:
        print('ålder NaNs', df.ålder.isna().sum())
    if not is_numeric_dtype(df.ålder):
        df['ålder'] = df.ålder.astype(int)

    # kön
    if df.kön.isna().sum() >0:
        print('kön NaNs', df.kön.isna().sum())
    
    # dist
    if not is_numeric_dtype(df.dist):
        df['dist'] = df.dist.astype('float')
    if df.dist.isna().sum() >0:
        print('dist NaNs', df.dist.isna().sum())
    
    # lopp_dist
    if not is_numeric_dtype(df.lopp_dist):
        df['lopp_dist'] = df.lopp_dist.str.replace('M','')
        df['lopp_dist'] = df.lopp_dist.astype('float')
    if df.lopp_dist.isna().sum() >0:
        print('lopp_dist NaNs', df.lopp_dist.isna().sum())
    
    # streck
    if not is_numeric_dtype(df.streck):
        df['streck'] = df.streck.str.replace('%','')
        df['streck'] = df.streck.replace(' ',None)
        df['streck'] = df.streck.astype('float')
    if df.streck.isna().sum() > 0:
        print(f'streck NaNs: {df.streck.isna().sum()}')
        
   # pris
    if not is_numeric_dtype(df.pris):
        df['pris'] = df.pris.astype('float')
    if df.pris.isna().sum() > 0:
        print(f'pris NaNs: {df.pris.isna().sum()}')

    ### Lagom längd på datum
    df['datum'] = df.datum.str.slice(0,10)

    ### skapa ny kolumn bins ###  
    före=len(df)      
    df = bins(df[df.dist > 1000])
    efter=len(df)
    if före-efter !=0:
        print('Tog bort',före-efter,'där dist <= 1000. Från',före,'till',efter)
        
    df.reset_index(drop=True, inplace=True)

##########################################  hhär följer all history ######################################
    if hist:
        print('start hist')
        ### hx_plac ###
        
        # diskad pga av galopp eller annat = 20
        # fullföljer men utanför prislistan = 15
        # byt '' till 15 (förr 13)
        df.loc[df.h1_plac == '', 'h1_plac'] = 15
        df.loc[df.h2_plac == '', 'h2_plac'] = 15
        df.loc[df.h3_plac == '', 'h3_plac'] = 15
        df.loc[df.h4_plac == '', 'h4_plac'] = 15
        df.loc[df.h5_plac == '', 'h5_plac'] = 15

        # byt '0' till 15 (förr 13)
        df.loc[df.h1_plac == '0', 'h1_plac'] = 15
        df.loc[df.h2_plac == '0', 'h2_plac'] = 15
        df.loc[df.h3_plac == '0', 'h3_plac'] = 15
        df.loc[df.h4_plac == '0', 'h4_plac'] = 15
        df.loc[df.h5_plac == '0', 'h5_plac'] = 15

        # byt 'd' till 20 (förr 14)
        df.loc[df.h1_plac == 'd', 'h1_plac'] = 20
        df.loc[df.h2_plac == 'd', 'h2_plac'] = 20
        df.loc[df.h3_plac == 'd', 'h3_plac'] = 20
        df.loc[df.h4_plac == 'd', 'h4_plac'] = 20
        df.loc[df.h5_plac == 'd', 'h5_plac'] = 20

        # ett 'r' för mycket
        df.loc[df.h1_plac == '0r', 'h1_plac'] = 15
        df.loc[df.h2_plac == '0r', 'h2_plac'] = 15
        df.loc[df.h3_plac == '0r', 'h3_plac'] = 15
        df.loc[df.h4_plac == '0r', 'h4_plac'] = 15
        df.loc[df.h5_plac == '0r', 'h5_plac'] = 15

        df.loc[df.h1_plac == '1r', 'h1_plac'] = 1
        df.loc[df.h2_plac == '1r', 'h2_plac'] = 1
        df.loc[df.h3_plac == '1r', 'h3_plac'] = 1
        df.loc[df.h4_plac == '1r', 'h4_plac'] = 1
        df.loc[df.h5_plac == '1r', 'h5_plac'] = 1

        df.loc[df.h1_plac == '2r', 'h1_plac'] = 2
        df.loc[df.h2_plac == '2r', 'h2_plac'] = 2
        df.loc[df.h3_plac == '2r', 'h3_plac'] = 2
        df.loc[df.h4_plac == '2r', 'h4_plac'] = 2
        df.loc[df.h5_plac == '2r', 'h5_plac'] = 2

        df.loc[df.h1_plac == '3r', 'h1_plac'] = 3
        df.loc[df.h2_plac == '3r', 'h2_plac'] = 3
        df.loc[df.h3_plac == '3r', 'h3_plac'] = 3
        df.loc[df.h4_plac == '3r', 'h4_plac'] = 3
        df.loc[df.h5_plac == '3r', 'h5_plac'] = 3

        df.loc[df.h1_plac == '4r', 'h1_plac'] = 4
        df.loc[df.h2_plac == '4r', 'h2_plac'] = 4
        df.loc[df.h3_plac == '4r', 'h3_plac'] = 4
        df.loc[df.h4_plac == '4r', 'h4_plac'] = 4
        df.loc[df.h5_plac == '4r', 'h5_plac'] = 4

        # byt 'k' till 15 (förr 13)
        df.loc[df.h1_plac == 'k','h1_plac'] = 15
        df.loc[df.h2_plac == 'k','h2_plac'] = 15
        df.loc[df.h3_plac == 'k','h3_plac'] = 15
        df.loc[df.h4_plac == 'k','h4_plac'] = 15
        df.loc[df.h5_plac == 'k','h5_plac'] = 15

        # byt 'p' till 15 (förr 13)
        df.loc[df.h1_plac == 'p','h1_plac'] = 15
        df.loc[df.h2_plac == 'p','h2_plac'] = 15
        df.loc[df.h3_plac == 'p','h3_plac'] = 15
        df.loc[df.h4_plac == 'p','h4_plac'] = 15
        df.loc[df.h5_plac == 'p','h5_plac'] = 15

        #### gör hx_plac till float ###
        df['h1_plac'] = df.h1_plac.astype('float')
        df['h2_plac'] = df.h2_plac.astype('float')
        df['h3_plac'] = df.h3_plac.astype('float')
        df['h4_plac'] = df.h3_plac.astype('float')
        df['h5_plac'] = df.h3_plac.astype('float')

        #### hx_odds ####
        # h1_odds
        if df.h1_odds.dtype == 'O':
            df.loc[df.h1_odds == '','h1_odds'] = np.NaN
            df.h1_odds = df['h1_odds'].str.replace(',', '.')
            df.loc[(df.h1_odds.str.contains('[a-z,-]',case=False)&df.h1_odds.notna() ),'h1_odds']=np.nan
            
        df['h1_odds'] = df.h1_odds.astype('float')

        # h2_odds
        if df.h2_odds.dtype == 'O':
            df.h2_odds = df['h2_odds'].str.replace(',', '.')
            df.loc[df.h2_odds == '','h2_odds'] = np.NaN
            df.loc[(df.h2_odds.str.contains('[a-z,-]',case=False)&df.h2_odds.notna() ),'h2_odds']=np.nan
            
        df['h2_odds'] = df.h2_odds.astype('float')

        # h3_odds
        if df.h3_odds.dtype == 'O':
            df.h3_odds = df['h3_odds'].str.replace(',', '.')
            df.loc[df.h3_odds == '','h3_odds'] = np.NaN
            df.loc[(df.h3_odds.str.contains('[a-z,-]',case=False)&df.h3_odds.notna() ),'h3_odds']=np.nan

        df['h3_odds'] = df.h3_odds.astype('float')

        # h4_odds
        if df.h4_odds.dtype == 'O':
            df.h4_odds = df['h4_odds'].str.replace(',', '.')
            df.loc[df.h4_odds == '','h4_odds'] = np.NaN
            df.loc[(df.h4_odds.str.contains('[a-z,-]',case=False)&df.h4_odds.notna() ),'h4_odds']=np.nan

        df['h4_odds'] = df.h4_odds.astype('float')

        # h5_odds
        if df.h5_odds.dtype == 'O':
            df.h5_odds = df['h5_odds'].str.replace(',', '.')
            df.loc[df.h5_odds == '','h5_odds'] = np.NaN
            df.loc[(df.h5_odds.str.contains('[a-z,-]',case=False)&df.h5_odds.notna() ),'h5_odds']=np.nan
   
        df['h5_odds'] = df.h5_odds.astype('float')

        # hx_spår '' sätts till NaN
        df.loc[df.h1_spår == '', 'h1_spår'] = np.nan
        df.loc[df.h2_spår == '', 'h2_spår'] =  np.nan
        df.loc[df.h3_spår == '', 'h3_spår'] =  np.nan
        df.loc[df.h4_spår == '', 'h4_spår'] =  np.nan
        df.loc[df.h5_spår == '', 'h5_spår'] =  np.nan
        # hx_spår ' ' sätts till NaN
        df.loc[df.h1_spår == ' ', 'h1_spår'] =  np.nan
        df.loc[df.h2_spår == ' ', 'h2_spår'] =  np.nan
        df.loc[df.h3_spår == ' ', 'h3_spår'] =  np.nan
        df.loc[df.h4_spår == ' ', 'h4_spår'] =  np.nan
        df.loc[df.h5_spår == ' ', 'h5_spår'] =  np.nan

        ### gör hx_spår numeriskt
        df.h1_spår = df.h1_spår.astype('float')
        df.h2_spår = df.h1_spår.astype('float')
        df.h3_spår = df.h1_spår.astype('float')
        df.h4_spår = df.h1_spår.astype('float')
        df.h5_spår = df.h1_spår.astype('float')

        ### gör hx_dist numeriskt ###
        df['h1_dist'] = df.h1_dist.astype('float')
        df['h2_dist'] = df.h2_dist.astype('float')
        df['h3_dist'] = df.h3_dist.astype('float')
        df['h4_dist'] = df.h4_dist.astype('float')
        df['h5_dist'] = df.h5_dist.astype('float')

        ### hx_kmtid och hx_auto ###
        a = df['h1_kmtid'].apply(lambda x: justkm(x))
        df['h1_kmtid'] = pd.DataFrame(item for item in a)[0]
        df['h1_auto'] = (pd.DataFrame(item for item in a)[1])*1
        a = df['h2_kmtid'].apply(lambda x: justkm(x))
        df['h2_kmtid'] = pd.DataFrame(item for item in a)[0]
        df['h2_auto'] = (pd.DataFrame(item for item in a)[1])*1
        a = df['h3_kmtid'].apply(lambda x: justkm(x))
        df['h3_kmtid'] = pd.DataFrame(item for item in a)[0]
        df['h3_auto'] = (pd.DataFrame(item for item in a)[1])*1
        a = df['h4_kmtid'].apply(lambda x: justkm(x))
        df['h4_kmtid'] = pd.DataFrame(item for item in a)[0]
        df['h4_auto'] = (pd.DataFrame(item for item in a)[1])*1
        a = df['h5_kmtid'].apply(lambda x: justkm(x))
        df['h5_kmtid'] = pd.DataFrame(item for item in a)[0]
        df['h5_auto'] = (pd.DataFrame(item for item in a)[1])*1

        ### hx_pris numeriskt ###
        def hx_pris(pr):
            pris = pr.copy()
            if pris.dtype != 'float':
                pris.loc[pris == ''] = None
                pris = pris.str.slice(stop=-1)
                pris = pris.astype('float')
            return pris
        
        df['h1_pris'] = hx_pris(df.h1_pris)
        df['h2_pris'] = hx_pris(df.h2_pris)
        df['h3_pris'] = hx_pris(df.h3_pris)
        df['h4_pris'] = hx_pris(df.h4_pris)
        df['h5_pris'] = hx_pris(df.h5_pris)
        
        if df.h1_dat.isna().sum()>0:
            print('h1_dat NaNs',df.h1_dat.isna().sum())

        # hx_dat från string till 10 chars (undvik datetime)
        def hx_dat_to_date(dat):
            #first remove '(' and ')'
            dat = dat.astype('string').str.replace('(','',regex=False)
            dat = dat.astype('string').str.replace(')','',regex=False)
            dat=dat.astype('O')
            dat[dat.isna()]=None
            dat = pd.to_datetime(dat,format='%y%m%d')
            # dat = dat.str.slice(0,10)
            return dat
        # transform all hx_dat    
        df.h1_dat=hx_dat_to_date(df.h1_dat)
        df.h2_dat=hx_dat_to_date(df.h2_dat)
        df.h3_dat=hx_dat_to_date(df.h3_dat)
        df.h4_dat=hx_dat_to_date(df.h4_dat)
        df.h5_dat=hx_dat_to_date(df.h5_dat)
        
        ### här byggs ny kolumner hx_perf och deltax
        # performance = prissumma*plac. Ju högre desto bättre
        def ber_perf(pris,plac):
            return np.sqrt(pris * np.exp(15-plac))

        df['h1_perf'] = ber_perf(df.h1_pris,df.h1_plac)
        df['h2_perf'] = ber_perf(df.h2_pris,df.h2_plac)
        df['h3_perf'] = ber_perf(df.h3_pris,df.h3_plac)
        df['h4_perf'] = ber_perf(df.h4_pris,df.h4_plac)
        df['h5_perf'] = ber_perf(df.h5_pris,df.h5_plac)

        # senast samt delat1-4
        def delta(dat1, dat2): # delta är dat1-det2
            dat1 = pd.to_datetime(dat1,format='%y%m%d')
            dat2 = pd.to_datetime(dat2,format='%y%m%d')
            delta= dat1-dat2
            return delta.dt.days.astype(float)
        
        datum = df.datum.copy() 
        datum = pd.to_datetime(datum,format='%Y-%m-%d')
        df['senast']=delta(datum, df.h1_dat.copy())
        df['delta1']=delta(df.h1_dat.copy(), df.h2_dat.copy())
        df['delta2']=delta(df.h2_dat.copy(), df.h3_dat.copy())
        df['delta3']=delta(df.h3_dat.copy(), df.h4_dat.copy())
        df['delta4']=delta(df.h4_dat.copy(), df.h5_dat.copy())
        
    print(f'klar med rensning avd={avd} df.shape={df.shape}')
    return df

# %%
