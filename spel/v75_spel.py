# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
from IPython import get_ipython

# %% 
# get_ipython().system('pip install -U streamlit')

# %%
import pandas as pd 
import numpy as np 
import streamlit as st
import sys

sys.path.append('C:\\Users\peter\\Documents\\MyProjects\\PyProj\\Trav\\spel\\modeller\\')
import V75_scraping as vs
import time
import logging
logging.basicConfig(filename='app.log', filemode='w', format='%(name)s - %(message)s',level=logging.INFO)

# %% [markdown]
# ## Do the scraping - loop över avdelningar

# %%
def do_scraping(driver_s, omg_links): #get data from web site
    print(omg_links)
    logging.basicConfig(filename='app.log', filemode='w', format='%(name)s - %(message)s',level=logging.INFO)
    
    vdict = {'datum':[], 'bana':[], 'avd':[], 'startnr':[],'häst':[],'ålder':[],'kön':[],'kusk':[],'lopp_dist':[],
    'start':[],'dist':[],'pris':[],'spår':[],'streck':[],'vodds':[],'podds':[],'kr':[], 
    'h1_dat':[],'h1_bana':[],'h1_kusk':[],'h1_plac':[],'h1_dist':[],'h1_spår':[],'h1_odds':[],'h1_pris':[],'h1_kmtid':[],
    'h2_dat':[],'h2_bana':[],'h2_kusk':[],'h2_plac':[],'h2_dist':[],'h2_spår':[],'h2_odds':[],'h2_pris':[],'h2_kmtid':[],
    'h3_dat':[],'h3_bana':[],'h3_kusk':[],'h3_plac':[],'h3_dist':[],'h3_spår':[],'h3_odds':[],'h3_pris':[],'h3_kmtid':[],
    'h4_dat':[],'h4_bana':[],'h4_kusk':[],'h4_plac':[],'h4_dist':[],'h4_spår':[],'h4_odds':[],'h4_pris':[],'h4_kmtid':[],
    'h5_dat':[],'h5_bana':[],'h5_kusk':[],'h5_plac':[],'h5_dist':[],'h5_spår':[],'h5_odds':[],'h5_pris':[],'h5_kmtid':[],
    }

    start_time = time.perf_counter()
    # Hela omgången (driver_s)
    dat = omg_links[0].split('spel/')[1][0:10]
    game_tab=driver_s.find_elements_by_class_name('game-table')[1:]                       ## alla lopp utan rubriker
    comb=driver_s.find_elements_by_class_name('race-combined-info')                 ## alla bana,dist,start
    priser=driver_s.find_elements_by_class_name('css-1lnkuf6-startlistraceinfodetails-styles--infoContainer')
    priser = [p.text for p in priser if 'Pris:' in p.text]                          ## alla lopps priser

    # ett lopp (de häst-relaterade som inte kan bli 'missing' tar jag direkt på loppnivå)
    for anr,avd in enumerate(game_tab):
        dat = omg_links[0].split('spel/')[1][0:10]  #gräv ut datum  
        logging.warning(dat+' avd: '+str(avd))

        bana=comb[anr].text.split('\n')[0]
        lopp_dist= comb[anr].text.split('\n')[1].split(' ')[0][:-1]
        start = comb[anr].text.split('\n')[1].split(' ')[1]

        pris=priser[0].split('-')[0].split(' ')[1]
        
        names     = avd.find_elements_by_class_name("horse-name")                         ## alla hästar/kön/åldet i loppet
        voddss    = avd.find_elements_by_class_name("vOdds-col")[1:]                      ## vodds i loppet utan rubrik
        poddss    = avd.find_elements_by_class_name("pOdds-col")[1:]                      ## podds i loppet utan rubrik
        rader     = avd.find_elements_by_class_name("startlist__row")                     ## alla rader i loppet
        strecks   = avd.find_elements_by_class_name("betDistribution-col")[1:]            ## streck i loppet  utan rubrik

        print('AVD', anr+1,bana,lopp_dist,start,end=' ')
        
        ## history ##
        hist=avd.find_elements_by_class_name("start-info-panel")  # all history för loppet

        for r,rad in enumerate(rader):
            # en häst
            logging.warning(r)
    
            vdict['datum'].append(dat)
            vdict['bana'].append(bana)
            vdict['avd'].append(anr+1)
            vdict['startnr'].append(r+1)
            vdict['häst'].append(names[r].text)
            vdict['start'].append(start)
            vdict['lopp_dist'].append(lopp_dist)
            vdict['pris'].append(pris)
            vdict['vodds'].append(voddss[r].text)            ##  vodds i lopp 1 utan rubrik
            vdict['podds'].append(poddss[r].text)            ##  podds i lopp 1 utan rubrik
            vdict['streck'].append(strecks[r].text)          ##  streck i lopp 1  utan rubrik
            ### kusk måste tas på hästnivå (pga att driver-col finns också i hist)
            vdict['kusk'].append(rad.find_elements_by_class_name("driver-col")[0].text)                         
            vdict['ålder'].append(rad.find_elements_by_class_name("horse-age")[0].text)                         
            vdict['kön'].append(rad.find_elements_by_class_name("horse-sex")[0].text)                          
            vdict['kr'].append(rad.find_elements_by_class_name("earningsPerStart-col")[0].text)  ##  kr/start i lopp 1 utan rubrik
            dist_sp = rad.find_elements_by_class_name("postPositionAndDistance-col")    ##  dist och spår i lopp 1 utan rubrik
            vdict['dist'].append(dist_sp[0].text.split(':')[0])
            vdict['spår'].append(dist_sp[0].text.split(':')[1])
            
            ### history
            h_dates =hist[r].find_elements_by_class_name('date-col')[1:]
            h_kuskar=hist[r].find_elements_by_class_name('driver-col')[1:]
            h_banor =hist[r].find_elements_by_class_name('track-col')[1:]
            h_plac  = hist[r].find_elements_by_class_name('place-col')[1:]  ## obs varannan rad (10 rader)
            
            h_dist = hist[r].find_elements_by_class_name('distance-col')[1:]
            h_spår = hist[r].find_elements_by_class_name('position-col')[1:]
            h_kmtid= hist[r].find_elements_by_class_name('kmTime-col')[1:]
            h_odds = hist[r].find_elements_by_class_name('odds-col')[1:]
            h_pris = hist[r].find_elements_by_class_name('firstPrize-col')[1:]
            
            for h,d in enumerate(h_dates)  : 
                fld='h'+str(h+1)+'_'
                vdict[fld+'dat'].append(d.text)
                vdict[fld+'bana'].append(h_banor[h].text)
                vdict[fld+'kusk'].append(h_kuskar[h].text)
                vdict[fld+'plac'].append(h_plac[h].text)
                vdict[fld+'dist'].append(h_dist[h].text)
                vdict[fld+'spår'].append(h_spår[h].text)
                vdict[fld+'kmtid'].append(h_kmtid[h].text)
                vdict[fld+'odds'].append(h_odds[h].text)
                vdict[fld+'pris'].append(h_pris[h].text)
            
            print('.',end='')
        print()
        
    print('\ndet tog',round(time.perf_counter() - start_time,3),'sekunder')
        
    return vdict
        

# %% [markdown]
# ## Start the scraping

# %%
global df
df=pd.DataFrame()
def scraping(omg_df, driver_s): #get data from web site
    global df 
    # global komplett
    #Läs in alla lopp från startlista
    komplett = do_scraping(driver_s, omg_df.Link)
    df = pd.DataFrame(komplett)
    print(df.shape)
    # df.columns=['datum','bana','avd','startnr','häst','kön','ålder', 'kusk', 'spår', 'streck','vodds', 'podds', 'pris', 'start','dist','lopp_dist','kr',  
    #     'h1_dat', 'h1_bana', 'h1_kusk', 'h1_plac', 'h1_dist','h1_spår','h1_kmtid','h1_odds','h1_pris',
    #     'h2_dat', 'h2_bana', 'h2_kusk', 'h2_plac', 'h2_dist','h2_spår','h2_kmtid','h2_odds','h2_pris',
    #     'h3_dat', 'h3_bana', 'h3_kusk', 'h3_plac', 'h3_dist','h3_spår','h3_kmtid','h3_odds','h3_pris',
    #     'h4_dat', 'h4_bana', 'h4_kusk', 'h4_plac', 'h4_dist','h4_spår','h4_kmtid','h4_odds','h4_pris',
    #     'h5_dat', 'h5_bana', 'h5_kusk', 'h5_plac', 'h5_dist','h5_spår','h5_kmtid','h5_odds','h5_pris'
    #     ]

    # fixa kolumner och NaN's 
    strukna = df[df.vodds=='EJ'][['datum','avd','startnr','häst','vodds']]
    df = df[df.vodds != 'EJ']  # Ta bort alla strukna
    df['pris']=df.pris.astype(float)
    import fixa_mer_features as ff2
    print(f"\n\ndf.shape före fixa_mer_features: {df.shape}\n")
    df = ff2.fixa_mer_features(df, True)
    print(f"\n\ndf.shape efter fixa_mer_features: {df.shape}")

    df.ålder = df.ålder.astype('int')
    
    # strecken
    print(f'streck före med NaNs: {df.streck.isna().sum()}')
    #df.streck = df.streck.str.replace('%','')
    #df.streck = df.streck.replace(' ',np.nan)
    df.streck = df.streck.astype('float')
    print(f'streck efter med NaNs: {df.streck.isna().sum()}')


    return df, strukna


# %%


# %% [markdown]
# ## Spara undan vodds och proba
# ## 30 min innan, 15 min innan, 5 min innan, efter att hela omgången är färdig
# 5 min innan är det faktiska spelet. Efter omgången har vi odds som inte var kända när kupongen lämnas in

# %%
def addera_vodds(df,vecko_df,dat,fas):
    # Uppdatera vecko_df 4 värden på vodds, prob_order
    #
    df.sort_values(by=['datum','avd','startnr','häst'],inplace=True)
    vecko_df.sort_values(by=['datum','avd','startnr','häst'],inplace=True)
    df.reset_index(drop=True,inplace=True)
    vecko_df.reset_index(drop=True,inplace=True)
    
    #### Check om utgått mellan varven
    vecko_sz = vecko_df[vecko_df.datum.str.slice(0,10)==dat].shape[0]
    df_sz =    df[df.datum==dat].shape[0]
    
    if vecko_sz>0 and vecko_sz != df_sz:
        print()
        print(f"OBS OBS! vecko_df har {vecko_sz} rader med df har {df_sz} rader. kolla att allt ok")
        vdf = vecko_df[vecko_df.datum.str.slice(0,10)==dat]
        print(vdf[~vdf.häst.isin(df.häst)][['datum','avd','startnr','häst']],'\n')

    # fas 1, 2, 3 innan första loppet. Fas 4 efter omgångens slut
    if fas==1:  # 15 minuter innan start
        # Ta bort ev rader med denna datum
        vecko_df=vecko_df[vecko_df.datum.str.slice(0,10)!=dat]

        # Lägg till efter sista raden
        vecko_df = pd.concat([vecko_df,df[['datum','avd','startnr','häst','vodds','prob_order']]])
        
    elif fas==2: # 10 min innan start
        # fyll i vodds2, proba_order2
        multiv = vecko_df.set_index(['datum', 'avd','startnr']).sort_index()
        multid = df.set_index(['datum', 'avd','startnr']).sort_index()
        
        multiv.vodds2.loc[dat] = multid.vodds
        multiv.prob_ord2.loc[dat] = multid.prob_order

        vecko_df = multiv.reset_index()
         
    elif fas==3: # 5 min innan start
        # fyll i vodds3, proba_order3
        multiv = vecko_df.set_index(['datum', 'avd','startnr']).sort_index()
        multid = df.set_index(['datum', 'avd','startnr']).sort_index()
        
        multiv.vodds3.loc[dat] = multid.vodds
        multiv.prob_ord3.loc[dat] = multid.prob_order

        vecko_df = multiv.reset_index()
 
    elif fas==4: # Efter omgången är klar och alla odds är de slutliga
        # fyll i vodds4, proba_order4
        multiv = vecko_df.set_index(['datum', 'avd','startnr']).sort_index()
        multid = df.set_index(['datum', 'avd','startnr']).sort_index()
        
        multiv.vodds4.loc[dat] = multid.vodds
        multiv.prob_ord4.loc[dat] = multid.prob_order

        vecko_df = multiv.reset_index()
 
    else:  # fel
        import sys
        sys.exit("FEL FEL FEL")
  

    cols = ['datum','avd','startnr', 'häst', 'vodds', 'prob_order', 'vodds2', 'prob_ord2', 'vodds3', 'prob_ord3', 'vodds4', 'prob_ord4']
    vecko_df = vecko_df.reindex(columns=cols)

    vecko_df.sort_values(by=['datum','avd','startnr','häst'],inplace=True)

    return vecko_df

# %% [markdown]
# ## Skapa V7-system

# %%
# SKAPA V75 RADER
def v75(df,modell, ant_rader=2):
    #,inplace=True)
    #df.reset_index(drop=True,inplace=True)

    #print(model.get_feature_importance(prettified=True))                    
    # selected_features = ['häst','kusk','bana','kön','spår','podds','dist','ålder','streck','vodds','bins','pris'] 
    selected_features = ['häst','kusk','bana','spår','bins','dist','ålder','streck','podds','vodds','pris'] 
    cat_features = ['häst','kusk','bana']

    proba = modell.predict_proba(df[selected_features])

    # Ordna proba per avdelning samt beräkna Kelly
    kassa=200
    df['proba'] = proba[:,1]
    df['f'] = (df.proba*df.vodds - 1) / (df.vodds-1)  # kelly formel
    df['spela'] = df.f >0
    df['insats'] = df.spela * df.f * kassa

    # Ta ut de 2 bästa per avd
    df.sort_values(['datum','avd','proba'],ascending=[True,True,False],inplace=True)
    proba_order=df.groupby(['datum','avd']).proba.cumcount()

    df['prob_order']=proba_order+1

    return(df)
    


# %%
def visa_v75(df,antal_rader):    
    df['p_o']=df.prob_order
    print(f"V75 {antal_rader}-raders")
    print(df[df.prob_order<=antal_rader][['avd','startnr','häst','vodds','p_o','proba','f','insats']])
    #print(df[df.prob_order<=antal_rader][['avd','startnr','häst','vodds','prob_order','proba','f','insats']])
    
def reserver(df, antal_rader):
    df['p_o']=df.prob_order
    print(f"Antal prob_order>{antal_rader} med insats>0")
    print(df[(df.prob_order>antal_rader) & (df.insats>0)] [['avd','startnr','häst','vodds','proba','p_o','f','insats']].sort_values(by='insats',ascending=False))
    df.drop('p_o',axis=1,inplace=True)

def visa_faser(vecko_df,dat,startnr=True):
    temp_df = vecko_df.copy()
    temp_df['X1'] = (temp_df.prob_order<=2)*1
    temp_df['X2'] = (temp_df.prob_ord2<=2)*1
    temp_df['X3'] = (temp_df.prob_ord3<=2)*1
    temp_df['X4'] = (temp_df.prob_ord4<=2)*1

    print('\nJämför faser')
    if startnr:
        print(temp_df[(temp_df.datum.str.slice(0,10) == dat) & ((temp_df.X1==1) | (temp_df.X2==1)| (temp_df.X3==1)| (temp_df.X4==1))][['avd','startnr','häst','vodds','vodds2','vodds3','vodds4','X1','X2','X3','X4']])
    else:
        print(temp_df[(temp_df.datum.str.slice(0,10) == dat) & ((temp_df.X1==1) | (temp_df.X2==1)| (temp_df.X3==1)| (temp_df.X4==1))][['avd','häst','vodds','vodds2','vodds3','vodds4','X1','X2','X3','X4']])            

# %% [markdown]
# # Main

# %%
##### Alternativ V75-rad #####
pd.set_option('display.width',100)
def alternativ_v75(FLAML_version=1):
    global df
    
    def remove_features(df,remove_mer=[]):
        df.drop(['avd','startnr','vodds','podds','bins','h1_dat','h2_dat','h3_dat','h4_dat','h5_dat'],axis=1,inplace=True) #
        if remove_mer:
            df.drop(remove_mer,axis=1,inplace=True)
        
        return df
        ######################################

    from catboost import CatBoostClassifier,Pool
    strukna = pd.DataFrame(columns=['datum','avd','startnr','häst','vodds'])

    # modell = 'modeller/model_senaste'
    # model = CatBoostClassifier()
    # model.load_model(modell, format='cbm')
    
    # Open the file in binary mode
    import pickle
    FLAML='FLAML'
    if FLAML_version==2:
        FLAML='FLAML2'
    with open('modeller\\'+FLAML+'_model.sav', 'rb') as file:
        # Call load method to deserialze
        print(FLAML)
        model = pickle.load(file)
  
    ### scraing och predict ###
    df, strukna = vs.v75_scraping(history=True,resultat=False) 

    dfr = remove_features(df.copy()) ## för cat_features och pool
    cat_features=list(dfr.loc[:,dfr.dtypes=='O'].columns)
    pool = Pool(dfr,cat_features=cat_features)
    # proba = model.predict_proba(pool)
    proba = model.predict_proba(dfr)

    # Ordna proba per avdelning samt beräkna Kelly
    kassa=200
    df['proba'] = proba[:,1]
    df['f'] = (df.proba*df.vodds - 1) / (df.vodds-1)  # kelly formel
    df['spela'] = df.f >0
    df['insats'] = df.spela * df.f * kassa

    # Ta ut de 2 bästa per avd
    df.sort_values(['datum','avd','proba'],ascending=[True,True,False],inplace=True)
    proba_order=df.groupby(['datum','avd']).proba.cumcount()

    df['prob_order']=proba_order+1

    visa_v75(df, 2)
    reserver(df,2)
    return df

# %%
def klassificering(df,proba=0.1,insats=0,antal_A=10):
    df_spel = df[(df.proba>proba)&(df.insats>insats)][['häst','avd','startnr','proba','prob_order','insats','streck','vodds']]

    antal = df_spel.shape[0]
    A_limit=df_spel.sort_values(by=['proba','avd','insats'],ascending=False).head(antal_A).proba.min()
    antal_B=antal-antal_A
    print('A_limit',round(A_limit,3))
    B_limit=df_spel.sort_values(by=['proba','avd','insats'],ascending=False).tail(antal_B).proba.min()
    print('B_limit',round(B_limit,3))

    print(f'Tot antal={len(df)}, antal_A={antal_A}, antal_B={antal_B}, antal_C={len(df)-(antal_A+antal_B)}')
    print()
    df_spel['ranking']='C'
    df_spel.loc[df_spel.proba>B_limit,'ranking'] = 'B'
    df_spel.loc[df_spel.proba>A_limit,'ranking'] = 'A'
    print(  'avd 1\n',df_spel[df_spel.avd==1][['häst','ranking','vann']])
    print('\navd 2\n',df_spel[df_spel.avd==2][['häst','ranking','vann']])
    print('\navd 3\n',df_spel[df_spel.avd==3][['häst','ranking','vann']])
    print('\navd 4\n',df_spel[df_spel.avd==4][['häst','ranking','vann']])
    print('\navd 5\n',df_spel[df_spel.avd==5][['häst','ranking','vann']])
    print('\navd 6\n',df_spel[df_spel.avd==6][['häst','ranking','vann']])
    print('\navd 7\n',df_spel[df_spel.avd==7][['häst','ranking','vann']])


    df['vann']=False
    # df.loc[(df.avd==1)&(df.startnr==3),'vann']=True
    # df.loc[(df.avd==2)&(df.startnr==2),'vann']=True
    # df.loc[(df.avd==3)&(df.startnr==4),'vann']=True
    # df.loc[(df.avd==4)&(df.startnr==5),'vann']=True
    # df.loc[(df.avd==5)&(df.startnr==10),'vann']=True
    # df.loc[(df.avd==6)&(df.startnr==4),'vann']=True
    # df.loc[(df.avd==7)&(df.startnr==4),'vann']=True
    print(f"\nant A rätt:{len(df[df.vann & (df.ranking=='A')])}, ant B rätt:{len(df[df.vann & (df.ranking=='B')])}")

    print('\n\nKandidater från Kelly (1 eller 2 får poäng=1 alt 4    eller tilldelas ranking=B) ')
    print(df[df.ranking!='A'].sort_values(by='insats',ascending=False)[['avd','häst','ranking','proba','insats']])

def sätt_poäng(df,extra_ettor=2,tvåor=4,treor=8,fyror=10):
    df['poäng'] = 15
    df.loc[df.prob_order==1,'poäng'] = 1
    df.sort_values(by='proba',ascending=False,inplace=True)
    ix=df[df.poäng!=1].iloc[:extra_ettor,:].index
    df.loc[ix,'poäng']=1
    ix=df[df.poäng!=1].iloc[:tvåor,:].index
    df.loc[ix,'poäng']=2
    ix=df[df.poäng>2].iloc[:treor,:].index
    df.loc[ix,'poäng']=3
    ix=df[df.poäng>3].iloc[:fyror,:].index
    df.loc[ix,'poäng']=4

    # min Kelly-häst
    df.sort_values(by='insats',ascending=False,inplace=True)
    ix=df[df.poäng>4].iloc[:1,:].index
    df.loc[ix,'poäng']=2
    print(f"\nmin Kelly-häst: {df.loc[ix,'häst'].values} avd={df.loc[ix,'avd'].values[0]} insats={round(df.loc[ix,'insats'].values[0],2)}\n")

    print('poängfördelning',df.poäng.value_counts())

    return df,ix

def uppdatera_FLAML(df_,version=1):
    df=df_.copy()
    import pickle
    FLAML='FLAML'
    if version==2:
        FLAML='FLAML2'
    with open('modeller\\'+FLAML+'_model.sav', 'rb') as file:
        # Call load method to deserialze
        # st.write(f'open {FLAML}_model_sav')
        model = pickle.load(file)

    proba = model.predict_proba(df.drop('startnr',axis=1))

    # Ordna proba per avdelning samt beräkna Kelly
    kassa=200
    df['proba'] = proba[:,1]
    df['f'] = (df.proba*df.vodds - 1) / (df.vodds-1)  # kelly formel
    df['spela'] = df.f >0
    df['insats'] = df.spela * df.f * kassa

    # Ta ut de 2 bästa per avd
    df.sort_values(['datum','avd','proba'],ascending=[True,True,False],inplace=True)
    proba_order=df.groupby(['datum','avd']).proba.cumcount()

    df['prob_order']=proba_order+1
    st.write(df.proba[0])
    return df
    
v75 = st.container()
scrape = st.container()
avd = st.container()
sortera = st.container()

with scrape:
    if st.button('scrape'):
        pd.DataFrame().to_csv('sparad_scrape.csv') # skriv över filen -> scrape börjar

def init():
    try:
        df=pd.read_csv('sparad_scrape.csv')
        dummy=df.bana.unique()  # finns data?
    except:    
        # data saknas
        scrape.write('Starta web-scraping för ny data')
        with st.spinner('Ta det lugnt!'):
            st.image('winning_horse.png')  # ,use_column_width=True)
            df=alternativ_v75(FLAML_version=1)
            st.balloons()
            
        df.to_csv('sparad_scrape.csv',index=False)
        # st.experimental_rerun()
    return df
    
with v75:
    omg_df = pd.read_csv('omg_att_spela_link.csv' )
    urlen=omg_df.Link.values[0]
    datum = urlen.split('spel/')[1][0:10]
    st.title('Veckans v75 -  ' +datum)
    df=init()            
    df,ix=sätt_poäng(df)
    df.rename(columns={"startnr":"nr"},inplace=True)  # För att få plats i två kolumner
    st.write(f"Kelly-häst:  Avd={df.loc[ix,'avd'].values[0]} \'{df.loc[ix,'häst'].values[0]}\' insats={round(df.loc[ix,'insats'].values[0],2)} odds={df.loc[ix,'vodds'].values[0]} streck={int(df.loc[ix,'streck'].values[0])}%")
   
with avd:
    use = avd.radio('Välj avdelning', ('Avd 1 och 2','Avd 3 och 4','Avd 5 och 6','Avd 7','exit'))
    avd.subheader(use)
    col1, col2 = st.columns(2)
    if use=='Avd 1 och 2':
        st.write(df.proba[0])
        col1.write(df[(df.avd==1)&(df.poäng<15)].sort_values(by=['poäng'])[['nr','häst','poäng']])
        col2.write(df[(df.avd==2)&(df.poäng<15)].sort_values(by=['poäng'])[['nr','häst','poäng']])
    elif use=='Avd 3 och 4':
        col1.write(df[(df.avd==3)&(df.poäng<15)].sort_values(by=['poäng'])[['nr','häst','poäng']])
        col2.write(df[(df.avd==4)&(df.poäng<15)].sort_values(by=['poäng'])[['nr','häst','poäng']])
    elif use=='Avd 5 och 6':
        col1.write(df[(df.avd==5)&(df.poäng<15)].sort_values(by=['poäng'])[['nr','häst','poäng']])
        col2.write(df[(df.avd==6)&(df.poäng<15)].sort_values(by=['poäng'])[['nr','häst','poäng']])
    elif use=='Avd 7':
        col1.write(df[(df.avd==7)&(df.poäng<15)].sort_values(by=['poäng'])[['nr','häst','poäng']])
    elif use=='exit':
        st.stop()    
    else:
        st.write('ej klart')

with sortera:   
    if st.sidebar.checkbox('se data'):
        sort=st.sidebar.radio('sortera på',['poäng','avd','insats'])
        if sort:
            if sort=='insats':
                st.write(df[['avd','häst','poäng','proba','insats','prob_order']].sort_values(by=['insats','proba'],ascending=[False,False]))
            else:
                st.write(df[['avd','häst','poäng','proba','insats','prob_order']].sort_values(by=[sort,'proba'],ascending=[True,False]))  
                  
if st.sidebar.checkbox('FLAML alt'):
    df.rename(columns={"nr":"startnr"},inplace=True)
    df=uppdatera_FLAML(df,version=2)
    df.to_csv('sparad_scrape.csv',index=False)
    df.rename(columns={"startnr":"nr"},inplace=True)  # För att få plats i två kolumner
else:
    df.rename(columns={"nr":"startnr"},inplace=True)
    df=uppdatera_FLAML(df,version=1)
    df.to_csv('sparad_scrape.csv',index=False)
    df.rename(columns={"startnr":"nr"},inplace=True)  # För att få plats i två kolumner
            
# print('\navd=2')
# print(df[(df.avd==2)&(df.poäng<15)].sort_values(by=['poäng'])[['startnr','häst','poäng']])

# print('\navd=3')
# print(df[(df.avd==3)&(df.poäng<15)].sort_values(by=['poäng'])[['startnr','häst','poäng']])

# print('\navd=4')
# print(df[(df.avd==4)&(df.poäng<15)].sort_values(by=['poäng'])[['startnr','häst','poäng']])

# print('\navd=5')
# print(df[(df.avd==5)&(df.poäng<15)].sort_values(by=['poäng'])[['startnr','häst','poäng']])

# print('\navd=6')
# print(df[(df.avd==6)&(df.poäng<15)].sort_values(by=['poäng'])[['startnr','häst','poäng']])

# print('\navd=7')
# print(df[(df.avd==7)&(df.poäng<15)].sort_values(by=['poäng'])[['startnr','häst','poäng']])

#%%