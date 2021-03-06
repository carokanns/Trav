# %%
from IPython import get_ipython

# %% [markdown]

# FÖRSÖK ATT GÖRA DEN SNABBARE MED THREADING
# # Gör scraping av en eller flera omgångar och returnerar en DataFrame
# - Körs med v75_scraping(resultat=False,history=True)
# - Parametrar: resultat=True/False, history=True/False
# - Input: omg_att_spela_link.csv() med en eler flera omgångar/datum

# %%
#!apt install chromium-chromedriver
# get_ipython().system('pip install selenium')

# %%
import pandas as pd
import numpy as np
import os

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException
from selenium.webdriver.chrome.options import Options

import sys
sys.path.append(
    'C:\\Users\\peter\\Documents\\MyProjects\\PyProj\\Trav\\spel\\')

import streamlit as st
import fixa_mer_features as ff2
import time

# %%
def get_webdriver(headless=True):
    # instance of Options class allows
    # us to configure Headless Chrome
    options = Options()

    if headless:
        # this parameter tells Chrome that
        # it should be run without UI (Headless)
        options.headless = True
        user_agent = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/95.0.4638.54 Safari/537.36'
        options.add_argument('user-agent={0}'.format(user_agent))
        options.add_argument("--window-size=1920x1080")


    # initializing webdriver for Chrome with our options
    driver = webdriver.Chrome(
        executable_path='C:\\Users\peter\\Documents\\MyProjects\\gecko\\chromedriver.exe', options=options)

    return driver

# %%
def städa_och_rensa(df, history):
    före = len(df)
    df = ff2.fixa_mer_features(df, history)
    efter = len(df)
    if före-efter != 0:
        print('rensade totalt bort', före-efter,
              'hästar i städa_och_rensa. Från', före, 'till', efter)
    return df

# %% utdelning
def scrape_utdelning(the_driver):
    # gammal lösn: utdelningar = the_driver.find_elements_by_class_name("css-mxas0-Payouts-styles--amount")

    utdelningar = the_driver.find_elements(By.CLASS_NAME, "css-fu45i8-Payouts-styles--amount")

    utd7 = utdelningar[0].text.replace(' ', '')
    utd7 = utd7.replace('kr', '')
    utd7 = utd7.replace('Jackpot', '0')

    utd6 = utdelningar[1].text.replace(' ', '')
    utd6 = utd6.replace('kr', '')
    utd6 = utd6.replace('Jackpot', '0')

    utd5 = utdelningar[2].text.replace(' ', '')
    utd5 = utd5.replace('kr', '')
    utd5 = utd5.replace('Jackpot', '0')

    return int(utd7), int(utd6), int(utd5)

# hämta utdelnings-info och spara på fil
def utdelning(driver_r, dat, bana):
    utd7, utd6, utd5 = scrape_utdelning(driver_r)
    assert utd7 > utd6 or utd7==0, '7 rätt skall ge mer pengar än 6 rätt'
    assert utd6 > utd5 or utd6 == 0, '6 rätt skall ge mer pengar än 5 rätt'

    print(f'utdelning: {utd7}, {utd6}, {utd5}')
    utd_file = 'C:\\Users\\peter\\Documents\\MyProjects\\PyProj\\Trav\\spel\\utdelning.csv'
    utdelning = pd.read_csv(utd_file)
    utd = pd.DataFrame([[dat, bana, utd7, utd6, utd5]])
    utd.columns = utdelning.columns
    utdelning = pd.concat([utdelning, utd], ignore_index=True)
    utdelning.drop_duplicates(['datum'], inplace=True)

    utdelning.sort_values(by=['datum'], inplace=True)
    utdelning.reset_index(drop=True, inplace=True)
    utdelning.to_csv(utd_file, index=False)

# %%
# returnerar en lista av placeringar i startnr-ordning
def inkludera_resultat(res_avd, anr):
    res_rader = res_avd.find_elements(By.CLASS_NAME,
        "startlist__row")  # alla rader i loppet
    res_startnr = res_avd.find_elements(By.CLASS_NAME,
        "css-1jc4209-horseview-styles--startNumber")  # alla startnr
    d = {'plac': [], 'startnr': []}
    for nr, rad in enumerate(res_rader):
        plac = rad.text.split(' ')[0]

        d['plac'].append(plac)
        d['startnr'].append(res_startnr[nr].text)

    temp_df = pd.DataFrame(d)
    temp_df['startnr'] = temp_df.startnr.astype(int)
    temp_df.sort_values(by='startnr', ascending=True, inplace=True)

    return temp_df.plac.to_list()

# %%

def en_rad(vdict, datum, bana, start, lopp_dist, avd, anr, r, rad, voddss, poddss, strecks, names, pris, history):
    # print('called with',anr, 'and', vdict['avd'])

    vdict['datum'].append(datum)
    vdict['bana'].append(bana)
    vdict['avd'].append(anr+1)
    vdict['startnr'].append(r+1)
    vdict['häst'].append(names[r].text)
    vdict['start'].append(start)
    vdict['lopp_dist'].append(lopp_dist)
    vdict['pris'].append(pris)
    vdict['vodds'].append(voddss[r].text)  # vodds i lopp 1 utan rubrik
    vdict['podds'].append(poddss[r].text)  # podds i lopp 1 utan rubrik
    # streck i lopp 1  utan rubrik
    vdict['streck'].append(strecks[r].text)
    # kusk måste tas på hästnivå (pga att driver-col finns också i hist)
    vdict['kusk'].append(
        rad.find_elements(By.CLASS_NAME, "driver-col")[0].text)
    vdict['ålder'].append(
        rad.find_elements(By.CLASS_NAME, "horse-age")[0].text)
    vdict['kön'].append(
        rad.find_elements(By.CLASS_NAME, "horse-sex")[0].text)

    vdict['kr'].append(rad.find_elements(
        By.CLASS_NAME, "earningsPerStart-col")[0].text)  # kr/start i lopp 1 utan rubrik

    # dist och spår i lopp 1 utan rubrik
    dist_sp = rad.find_elements(
        By.CLASS_NAME, "postPositionAndDistance-col")

    assert len(
        dist_sp) > 0, f'dist_sp ej len>0 len={len(dist_sp)} {dist_sp}'
    # print('rad',rad)

    # print(dist_sp)
    vdict['dist'].append(dist_sp[0].text.split(':')[0])
    vdict['spår'].append(dist_sp[0].text.split(':')[1])

    if history:
        ## history från startlistan ##
        hist = avd.find_elements(By.CLASS_NAME,
                                    "start-info-panel")  # all history för loppet

        h_dates = hist[r].find_elements(By.CLASS_NAME, 'date-col')[1:]
        h_kuskar = hist[r].find_elements(By.CLASS_NAME,
                                            'driver-col')[1:]
        h_banor = hist[r].find_elements(By.CLASS_NAME, 'track-col')[1:]
        h_plac = hist[r].find_elements(By.CLASS_NAME,
                                        'place-col')[1:]  # obs varannan rad (10 rader)
        h_plac = h_plac[::2]   # ta ut varannat värde

        h_dist = hist[r].find_elements(By.CLASS_NAME,
                                        'distance-col')[1:]
        h_spår = hist[r].find_elements(By.CLASS_NAME,
                                        'position-col')[1:]
        h_kmtid = hist[r].find_elements(
            By.CLASS_NAME, 'kmTime-col')[1:]
        h_odds = hist[r].find_elements(By.CLASS_NAME, 'odds-col')[1:]
        h_pris = hist[r].find_elements(By.CLASS_NAME,
                                        'firstPrize-col')[1:]

        for h, d in enumerate(h_dates):
            fld = 'h'+str(h+1)+'_'
            dtext = d.text
            vdict[fld+'dat'].append(dtext)
            vdict[fld+'bana'].append(h_banor[h].text)
            vdict[fld+'kusk'].append(h_kuskar[h].text)
            vdict[fld+'plac'].append(h_plac[h].text)
            vdict[fld+'dist'].append(h_dist[h].text)
            vdict[fld+'spår'].append(h_spår[h].text)
            vdict[fld+'kmtid'].append(h_kmtid[h].text)
            vdict[fld+'odds'].append(h_odds[h].text)
            vdict[fld+'pris'].append(h_pris[h].text)
        if len(h_dates) < 5:
            # Duplicera hist_4
            fld = 'h'+str(5)+'_'
            h = 3
            vdict[fld+'dat'].append(dtext)
            vdict[fld+'bana'].append(h_banor[h].text)
            vdict[fld+'kusk'].append(h_kuskar[h].text)
            vdict[fld+'plac'].append(h_plac[h].text)
            vdict[fld+'dist'].append(h_dist[h].text)
            vdict[fld+'spår'].append(h_spår[h].text)
            vdict[fld+'kmtid'].append(h_kmtid[h].text)
            vdict[fld+'odds'].append(h_odds[h].text)
            vdict[fld+'pris'].append(h_pris[h].text)
            
    # print('leaving with',anr, 'and',vdict['avd'])     
    return vdict


def do_scraping(driver_s, driver_r, avdelningar, history, datum):  # get data from web site
    # logging.basicConfig(filename='app.log', filemode='w',
    #                     format='%(name)s - %(message)s', level=logging.INFO)

    vdict = {'datum': [], 'bana': [], 'avd': [], 'startnr': [], 'häst': [], 'ålder': [], 'kön': [], 'kusk': [], 'lopp_dist': [],
             'start': [], 'dist': [], 'pris': [], 'spår': [], 'streck': [], 'vodds': [], 'podds': [], 'kr': [], }
    if driver_r:
        vdict.update({'plac': []})
    if history:
        vdict.update(
            {'h1_dat': [], 'h1_bana': [], 'h1_kusk': [], 'h1_plac': [], 'h1_dist': [], 'h1_spår': [], 'h1_odds': [], 'h1_pris': [], 'h1_kmtid': [],
             'h2_dat': [], 'h2_bana': [], 'h2_kusk': [], 'h2_plac': [], 'h2_dist': [], 'h2_spår': [], 'h2_odds': [], 'h2_pris': [], 'h2_kmtid': [],
             'h3_dat': [], 'h3_bana': [], 'h3_kusk': [], 'h3_plac': [], 'h3_dist': [], 'h3_spår': [], 'h3_odds': [], 'h3_pris': [], 'h3_kmtid': [],
             'h4_dat': [], 'h4_bana': [], 'h4_kusk': [], 'h4_plac': [], 'h4_dist': [], 'h4_spår': [], 'h4_odds': [], 'h4_pris': [], 'h4_kmtid': [],
             'h5_dat': [], 'h5_bana': [], 'h5_kusk': [], 'h5_plac': [], 'h5_dist': [], 'h5_spår': [], 'h5_odds': [], 'h5_pris': [], 'h5_kmtid': [],
             })

 
    if driver_r:
        result_tab = driver_r.find_elements(By.CLASS_NAME,
            'game-table')[:]  # alla lopp med resultatordning
        if len(result_tab) == 8:
            result_tab = result_tab[1:]
        print('ant resultat', len(result_tab))

    start_tab = driver_s.find_elements(By.CLASS_NAME, 'game-table')[:]  # alla lopp med startlistor
    if len(start_tab) == 8:
        start_tab = start_tab[1:]
    print('ant lopp', len(start_tab))

    comb = driver_s.find_elements(By.CLASS_NAME,'race-combined-info')  # alla bana,dist,start
    priselement = driver_s.find_elements(By.CLASS_NAME,
        'css-1lnkuf6-startlistraceinfodetails-styles--infoContainer')
    NOK, EUR = False, False
    for p in priselement:
        if 'EUR' in p.text:
            EUR = True
            break
        elif 'NOK' in p.text:
            NOK = True
            break
    print('EUR:', EUR, 'NOK:', NOK)
    # alla lopps priser
    priser = [p.text for p in priselement if 'Pris:' in p.text]

    if len(priser) == 0:
        priser = [p.text for p in priselement 
                  if 'åriga' not in p.text and 
                  ('Tillägg' not in p.text) and 
                  ('EUR' in p.text or 'NOK' in p.text)]  # alla lopps priser
    # print('priser', priser)
    print('Ant priser', len(priser))
    
    # ett lopp (de häst-relaterade som inte kan bli 'missing' tar jag direkt på loppnivå)
    for anr, avd in enumerate(start_tab):
        if anr+1 not in avdelningar:
            continue
        print('avd', anr+1)
        # logging.warning(datum+' avd: '+str(avd))

        bana = comb[anr].text.split('\n')[0]
        lopp_dist = comb[anr].text.split('\n')[1].split(' ')[0][:-1]
        start = comb[anr].text.split('\n')[1].split(' ')[1]

        if EUR:
            pris = priser[anr].split('-')[0] + '0'    # Euro -> SEK
        elif NOK:
            pris = priser[anr].split('-')[0].replace('.', '')  # NOK -> SEK
        else:
            pris = priser[anr].split('-')[0].split(' ')[1]
            
        pris=pris.replace(' ','')
        pris=pris.replace('.','')
        
        # print('pris:', pris)
        names = avd.find_elements(By.CLASS_NAME,
            "horse-name")  # alla hästar/kön/åldet i loppet
        voddss = avd.find_elements(By.CLASS_NAME,
            "vOdds-col")[1:]  # vodds i loppet utan rubrik
        poddss = avd.find_elements(By.CLASS_NAME,
            "pOdds-col")[1:]  # podds i loppet utan rubrik
        rader = avd.find_elements(By.CLASS_NAME,
            "startlist__row")  # alla rader i loppet
        strecks = avd.find_elements(By.CLASS_NAME,
            "betDistribution-col")[1:]  # streck i loppet  utan rubrik
        # print('ant names,vodds,podds,rader,streck', len(names), len(voddss), len(poddss), len(strecks))
        if driver_r:
            # resultat från resultatsidan
            res_avd = result_tab[anr]
            # placeringar sorterade efter startnr för en avd
            places = inkludera_resultat(res_avd, anr+1)
            # res_startnr = res_avd.find_elements(By.CLASS_NAME,"css-1jc4209-horseview-styles--startNumber")[1:]
            vdict['plac'] += places    # konkatenera listorna

        # print('AVD', anr+1, bana, lopp_dist, start, end=' ')
    
        # from concurrent.futures import ThreadPoolExecutor 
        # from concurrent.futures import as_completed
        
        # vdicts=[]
        # with ThreadPoolExecutor() as e:

        for r, rad in enumerate(rader):
            vdict = en_rad(vdict, datum, bana, start, lopp_dist, avd, anr, r, rad, voddss, poddss, strecks, names, pris, history)
            print('.', end='')
            
            # print(f'anr={anr+1} r={r+1}')
            # v = [e.submit(en_rad, vdict, datum, bana, start,lopp_dist, avd, anr, r, rad, voddss, poddss, strecks, names, pris, history)
            
            
        # vdict=v[0].result()
        # print(vdict)
    print()    
    
    
    return vdict, bana

# %% popup för vilken info som ska visas
def anpassa(driver_s):
    sl = driver_s.find_elements(By.CLASS_NAME, "startlist")
    # print('sl all text\n', sl[0].text)
    # print('sl end')
    buts = sl[0].find_elements(
        By.CLASS_NAME, "css-eugx3a-startlistoptionsview-styles--configButton-Button--buttonComponent")

    buts[0].click()
    
    time.sleep(1)
    
    print('klickade på', buts[0].text)

    # tics = driver_s.find_elements_by_class_name("css-1hngy38-Checkbox-styles--label")
    WebDriverWait(driver_s, 10).until(EC.presence_of_element_located((By.CLASS_NAME, 'css-1hngy38-Checkbox-styles--label')))
    tics = driver_s.find_elements(By.CLASS_NAME,"css-1hngy38-Checkbox-styles--label")
    driver_s.implicitly_wait(5)     # seconds
    # print('len tics',len(tics))

    flag1 = flag2 = flag3 = flag4 = flag5 = flag6 = flag7 = flag8 = flag9 = True
    for t in tics:
        if not (flag1 or flag2 or flag3 or flag4 or flag5 or flag6 or flag7 or flag8 or flag9):
            print('anpassa klar - break')
            break
        if t.text == '':
            continue

        if flag1 and t.text == 'VÄRMNINGS­VIDEO':
            t.click()
            # print('värmn')
            flag1 = False
        elif flag2 and t.text == 'UTÖKA ALLA STARTLISTOR SAMTIDIGT':
            t.click()
            # print('utöka')
            flag2 = False
        elif flag3 and t.text == 'TIPSKOMMENTARER':
            t.click()
            # print('tips')
            flag3 = False
        elif flag4 and ('LOPPKOMMENTARER' in t.text):
            t.click()
            # print('komment')
            flag4 = False
        elif flag5 and t.text == 'KR/START':
            t.click()
            # print('kr')
            flag5 = False
        elif flag6 and t.text == 'DISTANS OCH SPÅR':
            # t.click()
            # if t.is_enabled():
            #     print('distans och spår är enabled')

            # if t.is_displayed():
            #     print('distans och spår är displayed')
                
            if t.is_selected():
                print('distans och spår är redan valt')
                flag6 = False
            else:
                # print(t.text, 'ej selected ännu')
                pass

            # WebDriverWait(t, 10).until(EC.element_to_be_clickable(t), message='distans och spår gick inte att klicka på')
            WebDriverWait(t,10)
            t.click()
            if t.is_selected():
                # print(t.text+' är korrekt')
                flag6 = False
            else:
                # print(t.text+' är fel')
                flag6 = True
            
            # print('efter click distans och spår')

        elif flag7 and t.text == 'V-ODDS':
            # t.click()
            # print('hoppar över voods click (verkar vara förifyllt')
            flag7 = False
        elif flag8 and t.text == 'P-ODDS':
            t.click()
            # print('podds')
            flag8 = False
        elif flag9 and t.text == 'HÄSTENS KÖN & ÅLDER':
            t.click()
            # print('kön')
            flag9 = False

    # print('Prova name '+'checkbox-ageAndSex')
    chkbx = driver_s.find_elements(By.NAME, 'checkbox-ageAndSex')[0]
    if chkbx.is_enabled():
    #     print(chkbx.text +' är enabled')
        pass
    if chkbx.is_displayed():
    #     print(chkbx.text+' är displayed')
        pass
    if chkbx.is_selected():
    #     print(chkbx.text+' är korrekt valt')
        pass
    else:
        # print(chkbx.text, ' ej selected ännu')
        chkbx.click()
    if chkbx.is_selected():
        print(chkbx.text+' är korrekt')
    
    
    ## Tryck på Spara-knappen ##
    driver_s.implicitly_wait(5)     # seconds
    save_button = driver_s.find_elements(By.CSS_SELECTOR, "[ class^='css-1ix']")
     
    save_button[0].click()
    print('efter click Spara')


def v75_scraping(resultat=False, history=False, headless=True, driver_s=None, driver_r=None):
    pd.options.display.max_rows = 200
    import concurrent.futures
    start_time = time.perf_counter()

    avd_lists = [[1], [2], [3], [4], [5], [6], [7]]
    resultat = [resultat]*len(avd_lists)
    history = [history]*len(avd_lists)
    headless = [headless]*len(avd_lists)
    driver_s = [driver_s]*len(avd_lists)
    driver_r = [driver_r]*len(avd_lists)
    
    df = pd.DataFrame()
    with concurrent.futures.ThreadPoolExecutor() as executor:
        for temp_df in executor.map(v75_threads, resultat, history, headless, avd_lists, driver_s, driver_r):
            df = pd.concat([df, temp_df])

    df.sort_values(by=['datum', 'avd', 'startnr', ], inplace=True)
    
    print('\ndet tog', round(time.perf_counter() - start_time, 3), 'sekunder')
    
    return df

#%%
def v75_threads(resultat=False, history=False, headless=True, avdelningar=[1,2,3,4,5,6,7], driver_s=None, driver_r=None):
    ### Hela loopen med alla lopp i alla veckor i omg_df ###
    omg_df = pd.read_csv(
        'C:\\Users\\peter\\Documents\\MyProjects\\PyProj\\Trav\\spel\\omg_att_spela_link.csv')
    df = pd.DataFrame()

    if not driver_r:
        if resultat:
            driver_r = get_webdriver(headless)  # get web driver for results
        else:
            driver_r = None

    if not driver_s:
        driver_s = get_webdriver(headless)  # web driver för startlista

    for enum, omg in enumerate(omg_df.Link.values):
        print(f'omgång {enum+1}:', omg)
        datum = omg.split('spel/')[1][0:10]

        driver_s.get(omg)              # öppna startlista
        driver_s.implicitly_wait(5)     # seconds

        if enum == 0:  # första gången startlista
            # ok till första popup om kakor
            driver_s.find_element(By.ID,"onetrust-accept-btn-handler").click()
            driver_s.fullscreen_window()
            driver_s.implicitly_wait(5)     # seconds

            driver_s.find_elements(By.CLASS_NAME, "race-info-toggle")[1].click()  # prissummor mm
            driver_s.implicitly_wait(5)     # seconds

            anpassa(driver_s)

        # resultat
        if resultat:
            driver_r.get(omg+'/resultat')   # öppna resultat
            driver_r.implicitly_wait(5)     # seconds
            if enum == 0:
                # ok till första popup om kakor
                driver_r.find_element(By.ID,"onetrust-accept-btn-handler").click()
                driver_r.fullscreen_window()

        # scraping
        komplett, bana = do_scraping(driver_s, driver_r, avdelningar, history, datum)
        temp_df = pd.DataFrame(komplett)
        df = pd.concat([df, temp_df])

        # utdelning
        if resultat:
            utdelning(driver_r, datum, bana)  # utdelning för denna omgång

    # för att göra tester och debugging efteråt
    # global df_spara
    # df_spara = df.copy()
    ###
    driver_s.quit()
    if resultat:
        driver_r.quit()

    strukna = df.loc[df.vodds == 'EJ', :].copy()  # Ta bort all strukna

    df = städa_och_rensa(df, history)

    return df

if __name__ == '__main__':
    pd.options.display.max_rows = 200
    import concurrent.futures
    start_time = time.perf_counter()
    print('START GOING')
    def start_scrape(avdelningar):
        df = v75_threads(resultat=True, history=True, headless=True, avdelningar=avdelningar)
        print('STOP GOING')
        return df
    
    avd_lists=[[1],[2],[3],[4],[5],[6],[7]]
    df=pd.DataFrame()    
    with concurrent.futures.ThreadPoolExecutor() as executor:
        for temp_df in executor.map(start_scrape,avd_lists):
            df = pd.concat([df, temp_df])
            
    df.sort_values(by=['datum', 'avd', 'startnr', ], inplace=True)        
    print(df)
    print('\ndet tog', round(time.perf_counter() - start_time, 3), 'sekunder')
import datetime    
print('END', datetime.datetime.now().strftime("%H.%M.%S"))

# %%
