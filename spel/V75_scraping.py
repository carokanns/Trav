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
import logging
import pandas as pd
import numpy as np
import os

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.common.keys import Keys


import sys
sys.path.append(
    'C:\\Users\\peter\\Documents\\MyProjects\\PyProj\\Trav\\spel\\')

import streamlit as st
import fixa_mer_features as ff2
import time

# %%
EXECUTABLE_PATH = 'C:\\Users\\peter\\Documents\\MyProjects\\gecko\\chromedriver.exe'
USER_AGENT = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/95.0.4638.54 Safari/537.36'
WINDOW_SIZE = "1920x1080"


def log_print(text):
    """Skriver ut på loggen och gör en print samt returnerar strängen (för assert)"""
    logging.info(text)
    print(text)

    return text

def get_webdriver(res, headless=True):
    options = Options()
    options.headless = headless
    options.add_argument('user-agent={0}'.format(USER_AGENT))
    options.add_argument("--window-size={0}".format(WINDOW_SIZE))
    try:
        driver = webdriver.Chrome(
            executable_path=EXECUTABLE_PATH, options=options)
        driver.implicitly_wait(10)
        print(f'startade webdriver för {"resultat" if res else "startlista"}')
        return driver
    except Exception as e:
        log_print(f'Error: {e}')

def quit_drivers(driver_r, driver_s):
    if driver_s:
        driver_s.quit()
        log_print('quit driver_s klar')
    if driver_r:
        driver_r.quit()
        log_print('quit driver_r klar')
        
# %%
def städa_och_rensa(df, history):
    före = len(df)
    try:
        df = ff2.fixa_mer_features(df, history)
        efter = len(df)
    except:
        log_print('++++++++++något gick fel i fixa_mer_features++++++++++++')
        st.error('något gick fel i fixa_mer_features')
            
    if före-efter != 0:
        log_print(f'rensade totalt bort {före-efter} hästar i städa_och_rensa. Från {före} till {efter}')
    return df

# %% utdelning
def scrape_utdelning(the_driver):
    # gammal lösn: utdelningar = the_driver.find_elements_by_class_name("css-mxas0-Payouts-styles--amount")

    utdelningar = the_driver.find_elements(By.CLASS_NAME, "css-fu45i8-Payouts-styles--amount")

    utd7 = utdelningar[0].text.replace(' ', '')
    utd7 = utd7.replace('kr', '')
    utd7 = utd7.replace('Jackpot', '0')
    
    assert utd7.isdigit(), log_print(f'utdelning 7 rätt är inte ett tal: {utd7}')
    
    utd6 = utdelningar[1].text.replace(' ', '')
    utd6 = utd6.replace('kr', '')
    utd6 = utd6.replace('Jackpot', '0')
    assert utd6.isdigit(), log_print(f'utdelning 6 rätt är inte ett tal: {utd6}')
    
    utd5 = utdelningar[2].text.replace(' ', '')
    utd5 = utd5.replace('kr', '')
    utd5 = utd5.replace('Jackpot', '0')
    assert utd5.isdigit(), log_print(f'utdelning 5 rätt är inte ett tal: {utd5}')
    
    return int(utd7), int(utd6), int(utd5)

# hämta utdelnings-info och spara på fil
def utdelning(driver_r, dat, bana):
    utd7, utd6, utd5 = scrape_utdelning(driver_r)
    assert utd7 > utd6 or utd7==0, log_print('7 rätt skall ge mer pengar än 6 rätt')
    assert utd6 > utd5 or utd6 == 0, log_print('6 rätt skall ge mer pengar än 5 rätt')

    log_print(f'utdelning: {utd7}, {utd6}, {utd5}')
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
    log_print(f'en_rad() called with anr={anr} and vdict["avd"]={vdict["avd"]}')

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
    dist_sp = rad.find_elements(By.CLASS_NAME, "postPositionAndDistance-col")

    assert len(dist_sp) > 0, f'dist_sp ej len>0 len={len(dist_sp)} {dist_sp}'
    # print('rad',rad)

    # print(dist_sp)
    vdict['dist'].append(dist_sp[0].text.split(':')[0])
    vdict['spår'].append(dist_sp[0].text.split(':')[1])

    if history:
        log_print(f'history avd {anr+1} för startnr {r+1}')
        ## history från startlistan ##
        # print(f'history avd {anr+1} för startnr {r+1}')
        hist = avd.find_elements(By.CLASS_NAME, "start-info-panel")  # all history för loppet

        h_dates = hist[r].find_elements(By.CLASS_NAME, 'date-col')[1:]
        h_kuskar = hist[r].find_elements(By.CLASS_NAME,'driver-col')[1:]
        h_banor = hist[r].find_elements(By.CLASS_NAME, 'track-col')[1:]
        h_plac = hist[r].find_elements(By.CLASS_NAME, 'place-col')[1:]  # obs varannan rad (10 rader)
        h_plac = h_plac[::2]   # ta ut varannat värde

        h_dist = hist[r].find_elements(By.CLASS_NAME, 'distance-col')[1:]
        h_spår = hist[r].find_elements(By.CLASS_NAME, 'position-col')[1:]
        h_kmtid = hist[r].find_elements(By.CLASS_NAME, 'kmTime-col')[1:]
        h_odds = hist[r].find_elements(By.CLASS_NAME, 'odds-col')[1:]
        h_pris = hist[r].find_elements(By.CLASS_NAME, 'firstPrize-col')[1:]

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
            # Duplicera hist_x
            ln=len(h_dates)
            print('*********************************************')
            print(names[r].text)
            print('len h_dates', len(h_dates))
            print('h_banor', h_banor[len(h_banor)-1].text)
            for h in range(ln, 5):
                fld = 'h'+str(h+1)+'_'
                vdict[fld+'dat'].append(dtext)
                vdict[fld+'bana'].append(h_banor[ln-1].text)
                vdict[fld+'kusk'].append(h_kuskar[ln-1].text)
                vdict[fld+'plac'].append(h_plac[ln-1].text)
                vdict[fld+'dist'].append(h_dist[ln-1].text)
                vdict[fld+'spår'].append(h_spår[ln-1].text)
                vdict[fld+'kmtid'].append(h_kmtid[ln-1].text)
                vdict[fld+'odds'].append(h_odds[ln-1].text)
                vdict[fld+'pris'].append(h_pris[ln-1].text)    
                        
        log_print(f'klart history avd {anr+1} för startnr {r+1}')
    log_print(f'klart rad {r+1}')
    return vdict


def do_scraping(driver_s, driver_r, avdelningar, history, datum):  # get data from web site

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
        log_print(f'find driver_r game-table avd {avdelningar}')
        
        # WebDriverWait(driver_r, 10).until(
        #     EC.presence_of_element_located((By.CLASS_NAME, 'game-table')))
        # print('1r ############################### ')
        
        driver_r.implicitly_wait(10)     # seconds
        result_tab = driver_r.find_elements(
            By.CLASS_NAME, 'game-table')[:]  # alla lopp med resultatordning
        log_print(f'2r ############################### len result_tab = {len(result_tab)}')
        
        if len(result_tab) == 0:
            log_print('result_tab not found - try again')
            # driver_r.implicitly_wait(10)     # seconds
            # WebDriverWait(driver_r, 10).until(
            #     EC.presence_of_element_located((By.CLASS_NAME, 'game-table')))[:]
            
            driver_r.implicitly_wait(10)     # seconds
            result_tab = driver_r.find_elements(By.CLASS_NAME, 'game-table')[:]  # alla lopp med resultatordning
        
        
        if len(result_tab) == 8:
            result_tab = result_tab[1:]

        assert len(result_tab) == 7, log_print(f'################################ Antal resultat är fel i avd {avdelningar}: {len(result_tab)}')
        
        
    log_print(f'find driver_s game-table avd {avdelningar}')
    # driver_s.implicitly_wait(10)     # seconds
    # WebDriverWait(driver_s, 10).until(
    #     EC.presence_of_element_located((By.CLASS_NAME, 'game-table')))
    # print('1 ############################### ')
    
    driver_s.implicitly_wait(10)     # seconds
    start_tab = driver_s.find_elements(By.CLASS_NAME, 'game-table')[:]  # alla lopp med startlistor
    log_print(f'2 ############################### len start_tab = {len(start_tab)}')
            
    if len(start_tab) == 8:
        start_tab = start_tab[1:]

    assert len(start_tab) == 7, log_print(f'################################### Antal lopp är fel: {len(start_tab)} avd {avdelningar}')
    
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
    log_print(f'EUR: {EUR}, NOK: {NOK}')
    # alla lopps priser
    priser = [p.text for p in priselement if 'Pris:' in p.text]

    if len(priser) == 0:
        priser = [p.text for p in priselement 
                  if 'åriga' not in p.text and 
                  ('Tillägg' not in p.text) and 
                  ('EUR' in p.text or 'NOK' in p.text)]  # alla lopps priser
    
    assert len(priser) == 7, log_print(f'Antal priser är fel: {len(priser)} avd {avdelningar}')
    
    # ett lopp (de häst-relaterade som inte kan bli 'missing' tar jag direkt på loppnivå)
    for anr, avd in enumerate(start_tab):
        if anr+1 not in avdelningar:
            continue
        assert anr+1 == avdelningar[0], log_print(f'Start avd {anr+1} ej samma som avdelningar[0] {avdelningar[0]}')

        bana = comb[anr].text.split('\n')[0]
        assert len(bana) > 0, f'Bana saknas: {bana} avd {avdelningar}'
        lopp_dist = comb[anr].text.split('\n')[1].split(' ')[0][:-1]
        assert len(lopp_dist) > 0, f'Loppdist saknas: {lopp_dist} avd {avdelningar}'
        start = comb[anr].text.split('\n')[1].split(' ')[1]
        assert len(start) > 0, f'Start saknas: {start} avd {avdelningar}'
        if EUR:
            pris = priser[anr].split('-')[0] + '0'    # Euro -> SEK
        elif NOK:
            pris = priser[anr].split('-')[0].replace('.', '')  # NOK -> SEK
        else:
            pris = priser[anr].split('-')[0].split(' ')[1]
            
        pris=pris.replace(' ','')
        pris=pris.replace('.','')
        assert len(pris) > 0, f'Pris saknas: {pris} avd {avdelningar}'
        
    
        names = avd.find_elements(By.CLASS_NAME, "horse-name")  # alla hästar/kön/åldet i loppet
        assert len(names) > 0, 'no names found avd {avdelningar}'
        voddss = avd.find_elements(By.CLASS_NAME, "vOdds-col")[1:]  # vodds i loppet utan rubrik
        assert len(voddss) > 0, 'no vodds found avd {avdelningar}'
        poddss = avd.find_elements(By.CLASS_NAME, "pOdds-col")[1:]  # podds i loppet utan rubrik
        assert len(poddss) > 0, 'no podds found avd {avdelningar}'
        rader = avd.find_elements(By.CLASS_NAME, "startlist__row")  # alla rader i loppet
        assert len(rader) > 0, 'no rows found   avd {avdelningar}'
        strecks = avd.find_elements(By.CLASS_NAME, "betDistribution-col")[1:]  # streck i loppet  utan rubrik
        assert len(strecks) > 0, 'no strecks found avd {avdelningar}'
        log_print(f'ant i avd {avdelningar}: names={len(names)}, vodds={len(voddss)}, podds={len(poddss)}, rader={len(rader)}, streck={len(strecks)}')
      
        if driver_r:
            # resultat från resultatsidan
            res_avd = result_tab[anr]
            assert res_avd, f'no result found: avd={anr+1}'
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
        log_print(f'Klar avd {anr+1}')    
    
    
    return vdict, bana



# %% popup för vilken info som ska visas
def anpassa(driver_s, avd):

    ###########   Lokal funktion   ########################
    def click_element(element_namn, element, avd):
        log_print(f'Väntade på {element_namn} avd={avd}')

        is_checkbox = driver_s.execute_script(
            "return arguments[0].type === 'checkbox';", element)
        log_print(f'{element_namn}.is_checkbox={is_checkbox} avdelning {avd}')
        log_print(f'{element_namn}.text={element.text}')
        actions = ActionChains(driver_s)
        actions.move_to_element(element).click().perform()
        is_selected = driver_s.execute_script(
            "return arguments[0].checked;", element)
        arg_selected = driver_s.execute_script(
            "return arguments[0].getAttribute('checked');", element)
        if not is_selected:
            log_print(f'klickar nu på {element_namn} avdelning {avd}')
            element.click()
        log_print(
            f'{element_namn}.is_selected={element.is_selected()} avdelning {avd}')

        # is_selected = driver_s.execute_script(
        #     "return arguments[0].checked;", element)
        # arg_selected = driver_s.execute_script(
        #     "return arguments[0].getAttribute('checked');", element)

        # log_print(f'is_selected={is_selected} avdelning {avd}')
        # log_print(f'arg_selected={arg_selected} avdelning {avd}')
        # if is_selected != arg_selected:
        #     log_print(
        #         f'is_selected={is_selected} != arg_selected={arg_selected} avdelning {avd}')

        if not is_selected:
            log_print(f'{element_namn} är ännu inte vald avdelning {avd}')
            driver_s.execute_script("arguments[0].click();", element)

        return True
    ########### Slut lokal funktion ########################

    avd = avd[0]
    wait = WebDriverWait(driver_s, 10)
    driver_s.implicitly_wait(10)
    
    sl = driver_s.find_elements(
        By.CLASS_NAME, "css-8y0fv8-Startlists-styles--startlistelement")  # täcker hela sidan för avd 1-7
    
    log_print(f'Hitta de tre knapparna "Andra spel", "Utöka alla" och "Anpassa" avd={avd}')
    avd_buttons = sl[avd-1].find_elements(
        By.CSS_SELECTOR, "button[class^='MuiButtonBase-root MuiButton-root']")

    assert len(avd_buttons) > 0, log_print(f'avd_buttons skall inte vara tom: {avd_buttons} för avd {avd}')

    log_print(f' avd_buttons[0].text  {avd_buttons[0].text}, avd_buttons[1]. text {avd_buttons[1].text}, avd_buttons[2].text {avd_buttons[2].text}')

    # Först utöka alla history för avdelningen
    log_print(f'Välj knappen "Utöka alla" avd={avd}')
    utöka_button = avd_buttons[1]
    log_print(f' Klickar nu på {utöka_button.text}, avdelning {avd}')
    actions = ActionChains(driver_s)
    actions.move_to_element(utöka_button).click().perform()
  
    log_print(f' Klickar nu med JavaScript på {utöka_button.text}, avdelning {avd}')
    driver_s.execute_script("arguments[0].click();", utöka_button)
    
    # Anpassa vilka kolumner i loppet som skall visas
    log_print(f'Välj knappen "Anpassa" avd={avd}')
    anpassa_button = avd_buttons[2]
    log_print(f' Klickar nu på {anpassa_button.text}, avdelning {avd}')
    actions = ActionChains(driver_s)
    actions.move_to_element(anpassa_button).click().perform()
    log_print(f' Klickar nu med JavaScript på {anpassa_button.text}, avdelning {avd}')
    driver_s.execute_script("arguments[0].click();", anpassa_button)

    # Hitta nästa huvud-element "Inför loppet" och vänta tills det är klickbart
    pre_race_element = wait.until(EC.element_to_be_clickable(
        (By.CSS_SELECTOR, '[data-test-id="desktop-category-pre-race"]')))
  
    log_print(f'klickar nu på huvud-elementet "Inför loppet". Avd={avd}')
    pre_race_element.click()
    
    ### Hitta nästa huvud element Hästinfo ###
    horse_info_element = wait.until(EC.element_to_be_clickable(
        (By.CSS_SELECTOR, '[data-test-id="desktop-category-horse-info"]')))
    log_print(f'klickar nu på huvud-elementet Hästinfo. Avdelning {avd}')
    horse_info_element.click()
    
    # Checkboxes under Hästinfo
    age_sex_checkbox = wait.until(EC.presence_of_element_located(
        (By.XPATH, "//div[@data-test-id='desktop-checkbox-ageAndSex']")))
   
    click_element('age_sex_checkbox',age_sex_checkbox,avd)
    
    ### Hitta nästa huvud-element 'Inför loppet' ###  
    pre_race_button = wait.until(EC.presence_of_element_located(
        (By.XPATH, "//div[@data-test-id='desktop-category-pre-race']")))
    log_print(f'klickar nu på huvud-elementet "inför loppet" avd {avd}')
    pre_race_button.click()
    
    # Checkboxes under 'Inför loppet' 
    logging.info(f'Letar nu upp checkboxen för vOdds avdelning {avd}')
    vOdds_checkbox = wait.until(EC.presence_of_element_located(
        (By.XPATH, "//div[@data-test-id='desktop-checkbox-vOdds']")))
    
    vOdds_checkbox.click() # Varför måste detta göras för enbart vOdds?
    click_element('vOdds_checkbox',vOdds_checkbox,avd)
        
    log_print(f'Letar nu upp checkboxen för pOdds avdelning {avd}')     
    pOdds_checkbox=wait.until(EC.presence_of_element_located(
            (By.XPATH, "//div[@data-test-id='desktop-checkbox-pOdds']")))
    
    click_element('pOdds_checkbox', pOdds_checkbox, avd) 

    log_print(f'Letar nu upp checkboxen för Distans & spår avdelning {avd}')
    dist_spår_checkbox = wait.until(EC.presence_of_element_located(
        (By.XPATH, "//div[contains(@class, 'MuiButtonBase-root MuiListItemButton-root')]//div[text()='Distans & spår']")))
    
    click_element('dist_spår_checkbox', dist_spår_checkbox, avd)
    
    ### Hitta nästa huvud-element 'Tidigare resultat' ###
    tidigare_res_button = wait.until(EC.presence_of_element_located(
        (By.XPATH, "//div[@data-test-id='desktop-category-previous-results']")))
    log_print(f'klickar nu på huvud-elementet "Tidigare resultat" avd {avd}')
    tidigare_res_button.click()
    
    # Checkboxes under 'Tidigare resultat' 
    log_print(f'Letar nu upp checkboxen för Kr/start avdelning {avd}')
    krPerStart_checkbox = wait.until(EC.presence_of_element_located(
        (By.XPATH, "//div[@data-test-id='desktop-checkbox-earningsPerStart']")))

    click_element('krPerStart_checkbox', krPerStart_checkbox, avd)
    
    ### Hitta nästa huvud-element 'Form & tips' ###
    form_tips = wait.until(EC.presence_of_element_located(
        (By.XPATH, "//div[@data-test-id ='desktop-category-form-and-tips']")))
    log_print(f'klickar nu på huvud-elementet "Tidigare resultat" avd {avd}')
    form_tips.click()
    
    log_print(f'I "Form & tips" skall allt vara släckt avd {avd}')
     
    # Checkboxes under 'Form och tips' 
    log_print(f'Letar nu upp checkboxen för Tipskommentar avd {avd}')
    tipsCom_checkbox = wait.until(EC.presence_of_element_located(
        (By.XPATH, "//div[@data-test-id='desktop-checkbox-showDagensSpelComment']")))
    click_element('tipsCom_checkbox', tipsCom_checkbox, avd)
     
    log_print(f'Letar nu upp checkboxen för Tipskommentar avd {avd}')
    statsCom_checkbox = wait.until(EC.presence_of_element_located(
        (By.XPATH, "//div[@data-test-id='desktop-checkbox-showTRMediaComment']")))
    click_element('statsCom_checkbox', statsCom_checkbox, avd)
     
    log_print(f'Letar nu upp checkboxen för Tipskommentar avd {avd}')
    loppCom_checkbox = wait.until(EC.presence_of_element_located(
        (By.XPATH, "//div[@data-test-id='desktop-checkbox-showRaceComments']")))
    click_element('loppCom_checkbox', loppCom_checkbox, avd)
    
    ### Tryck på Aktivera-knappen Dvs spara inställningarna ###
    log_print(f"Leta upp Aktivera knappen avd {avd}") 
    Aktivera_button = wait.until(EC.presence_of_element_located(
        (By.XPATH, "//button[@data-test-id='desktop-button-activate']")))

    Aktivera_button.click()
    log_print(f'Klickade på Aktivera. Färdig med avd={avd}')
        

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
def v75_threads(resultat=False, history=False, headless=True, avdelningar=None, driver_s=None, driver_r=None):
    ### Hela loopen med alla lopp i alla veckor i omg_df ###
    print('avdelningar', avdelningar)
    omg_df = pd.read_csv(
        'C:\\Users\\peter\\Documents\\MyProjects\\PyProj\\Trav\\spel\\omg_att_spela_link.csv')

    if not driver_r:
        if resultat:
            print('startar webdriver driver_r för avd', avdelningar)
            driver_r = get_webdriver(True,headless)  # get web driver for results
            print('Startade driver_r för avd', avdelningar)
        else:
            driver_r = None

    if not driver_s:
        print('Startar webdriver driver_s för avd', avdelningar)
        driver_s = get_webdriver(False,headless)  # web driver för startlista
        print('Startade driver_s för avd', avdelningar)

    for enum, omg in enumerate(omg_df.Link.values):
        print(f'omgång {enum+1}:', omg)
        datum = omg.split('spel/')[1][0:10]
        df = pd.DataFrame()
        print('Öppnar startlist-sidan med driver_s för avd', avdelningar)
        print('omg', omg)
        
        # _=input('Tryck enter för att öppna startlistan\n')
        driver_s.implicitly_wait(10)     # seconds
        driver_s.get(omg)              # öppna startlista
        
        print('############## Öppnade startlist-sidan med driver_s för avd', avdelningar)

        if enum == 0:  # första gången startlista
            # ok till första popup om kakor
            
            print('find cookie popup')
            # _ = input('Tryck enter för find cookie popup\n')
            driver_s.implicitly_wait(20)  # seconds
            but_kakor = driver_s.find_element(By.ID,"onetrust-accept-btn-handler")
            but_kakor_text = but_kakor.text
            
            print('klickar på ', but_kakor_text, 'för avd', avdelningar)
            # _ = input(f'Tryck enter för att klicka på {but_kakor_text}\n')
            
            driver_s.implicitly_wait(10)     # seconds
            but_kakor.click()
            driver_s.fullscreen_window()
            
            print( 'klickade på ', but_kakor_text, 'för avd', avdelningar)
            
            # _ = input('Tryck enter för att klicka bort reklamen\n')
            
            # try:
            #     driver_s.implicitly_wait(1)     # seconds
            #     driver_s.find_element(By.CLASS_NAME, "css-1m9bhvf-Typography-styles--body1-Typography--Typography").click()
            #     print('********Så länge reklam för vm i travspel finns kvar gör vi Try hät')
            # except:
            #     print("*********Är VM i travspel bortplockat nu? Ta bort denn kod i så fall.")
            #     pass
            
            # _ =input('Efter Try: Klicka för find lopp-info\n')
            driver_s.implicitly_wait(20)  # seconds
            race_info = driver_s.find_elements(By.CLASS_NAME, "race-info-toggle")
            if len(race_info)==0: #Try again
                log_print(f'try race_info igen för avd {avdelningar}')
                input('enter för att försöka igen med race_info')
                # WebDriverWait(driver_s, 10).until(
                #     EC.presence_of_element_located((By.CLASS_NAME, "race-info-toggle")))
                
                driver_s.implicitly_wait(10)  # seconds
                race_info = driver_s.find_elements(By.CLASS_NAME, "race-info-toggle")
                
            print(f'len(race_info)= {len(race_info)} för avd {avdelningar} resultat={resultat}')
            assert len(race_info) > 0, f'race_info måste innehålla en info per lopp men är tom, avd={avdelningar} resultat={resultat}' 
            
            driver_s.find_elements(By.CLASS_NAME, "race-info-toggle")[1].click()  # prissummor mm
            
            # _ = input('Efter prissummor mm: Klicka på OK för att fortsätta\n'  )
            
            print('Kör "anpassa" med driver_s för avd', avdelningar)
            anpassa(driver_s,avdelningar)
            print('Klar "anpassa" med driver_s för avd', avdelningar)
            # _ = input('Efter anpassa: Klicka på OK för att fortsätta\n'  )
            
        # resultat
        if resultat:
            print('Öppnar resultatsidan med driver_r för avd', avdelningar)
            driver_r.implicitly_wait(10) # seconds
            driver_r.get(omg+'/resultat')   # öppna resultat
    
            print('Öppnade resultatsidan med driver_r för avd', avdelningar)
            # _ = input('Efter öppnade resultatlistan: Klicka på OK för att fortsätta\n'  )
            if enum == 0:
                # ok till första popup om kakor
                # WebDriverWait(driver_r, 10).until(
                #     EC.presence_of_element_located((By.ID, "onetrust-accept-btn-handler")))
                
                driver_r.implicitly_wait(10)  # seconds
                but_kakor = driver_r.find_element(By.ID,"onetrust-accept-btn-handler")
                driver_r.implicitly_wait(10)  # seconds
                but_kakor.click()
                driver_r.fullscreen_window()
                
                # try:
                #     driver_r.implicitly_wait(1)     # seconds
                #     driver_r.find_element(By.CLASS_NAME, "css-1m9bhvf-Typography-styles--body1-Typography--Typography").click()
                #     print('******* (resultat) Så länge reklam för vm i travspel finns kvar')   
                # except:
                #     print("********Är VM i travspel bortplockat nu?")
                #     pass

        # scraping
        log_print(f'Kör scraping med driver_s och driver_r för avd {avdelningar}')
        try:
            komplett, bana = do_scraping(driver_s, driver_r, avdelningar, history, datum)
            log_print(f'Klar scraping med driver_s och driver_r för avd {avdelningar}')
        except:
            log_print(f'************************** Något gick fel i do_scraping avd {avdelningar} - quit drive *******************************') 
            quit_drivers(driver_s, driver_r)    
            log_print(f'df.shape={df.shape}')
            return df
        
        temp_df = pd.DataFrame(komplett)
        df = pd.concat([df, temp_df])

        # utdelning
        if resultat:
            try:
                print('Kör utdelning med driver_r för avd', avdelningar)
                utdelning(driver_r, datum, bana)  # utdelning för denna omgång
                print('Klar utdelning med driver_r för avd', avdelningar)

            except:
                print(f'************************** Något gick fel i utdelning(...) avd {avdelningar} - quit drivers *******************************') 
                quit_drivers(driver_s, driver_r)    
                return df
            
    # för att göra tester och debugging efteråt
    # global df_spara
    # df_spara = df.copy()
    ###
    
    quit_drivers(driver_s, driver_r)
    
    strukna = df.loc[df.vodds == 'EJ', :].copy()  # Ta bort all strukna

    df = städa_och_rensa(df, history)

    return df

if __name__ == '__main__':
    """ Här kan vi köra direkt från denna py-fil
    """
    
    logging.basicConfig(level=logging.INFO, filemode='a', filename='v75.log', force=True,
                    encoding='utf-8', format='v75:' '%(asctime)s - %(levelname)s - %(lineno)d - %(message)s')
    logging.info('Startar v75_scraping.py')
    
    pd.options.display.max_rows = 200
    import concurrent.futures
    start_time = time.perf_counter()
    print('START GOING')
    
    ######### settings #########
    omg_df = pd.read_csv(
        'C:\\Users\\peter\\Documents\\MyProjects\\PyProj\\Trav\\spel\\omg_att_spela_link.csv')
    # avd_list=[[1],[2],[3],[4],[5],[6],[7]]
    # avd_list=[[1],[2],[3],[4],[5],[6]]
    # avd_list=[[1],[2],[3],[4],[5]]
    # avd_list=[[1],[2],[3],[4]]
    avd_list=[[1],[2],[3]]
    # avd_list=[[1],[2]]
    # avd_list=[[1]]
    concurrency=False
    resultat=False
    history=True
    headless=False
    print(f'Kör med avd_list={avd_list}, resultat={resultat}, history={history}, headless={headless}, omg={omg_df.Link.values}')
    
    def start_scrape(avd_list):
        df = v75_threads(resultat=resultat, history=history, headless=headless, avdelningar=avd_list)
        print('STOP GOING')
        return df
    
    # df_list = ['df1','df2','df3','df4','df5','df6','df7']
    
    df=pd.DataFrame()

    if concurrency:
        with concurrent.futures.ThreadPoolExecutor() as executor:
            results = executor.map(start_scrape, avd_list)
            for i, result in enumerate(results):
                # df_list[i] = result
                df = pd.concat([df, result])
    else:            
        for enum,avd in enumerate(avd_list):
            df = pd.concat([df,start_scrape(avd)], ignore_index=True)
            
    log_print(f'efter allt klart: df.shape={df.shape}')    
    df.sort_values(by=['datum', 'avd', 'startnr', ], inplace=True)        
    print(df)
    log_print(f'\ndet tog {round(time.perf_counter() - start_time, 3)} sekunder')
    
import datetime    
print('END', datetime.datetime.now().strftime("%H.%M.%S"))

# %%
