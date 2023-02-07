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


def log_print(text, logging_level='d'):
    """Skriver ut på loggen och gör en print samt returnerar strängen (för assert)"""
    if logging_level == 'd':
        logging.debug(text)
    else:     
        if logging_level == 'i':
            logging.info(text)
        elif logging_level == 'w':
            logging.warning(text)
        elif logging_level == 'e':
            logging.error(text)
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
        log_print(f'startade webdriver för {"resultat" if res else "startlista"}','i')
        return driver
    except Exception as e:
        log_print(f'Error: {e}','e')

def quit_drivers(driver_r, driver_s):
    if driver_s:
        driver_s.quit()
        log_print(f'quit driver_s klar','i')
    if driver_r:
        driver_r.quit()
        log_print(f'quit driver_r klar','i')
        
# %%
def städa_och_rensa(df, history):
    före = len(df)
    efter = före
    # df.to_csv('temp.csv', index=False)
    try:
        df = ff2.fixa_mer_features(df, history)
        efter = len(df)
    except:
        log_print(f'++++++++++något gick fel i fixa_mer_features++++++++++++','e')
        st.error('något gick fel i fixa_mer_features')
            
    if före-efter != 0:
        log_print(f'rensade totalt bort {före-efter} hästar i städa_och_rensa. Från {före} till {efter}','i')
    return df

# %% utdelning
def scrape_utdelning(the_driver):
    
    elem = the_driver.find_element_by_xpath("//span[text()='Utdelning 7 rätt']")
    utd7 = elem.find_element_by_xpath("following-sibling::span").text

    utd7 = utd7.replace(' ', '')
    utd7 = utd7.replace('kr', '')
    utd7 = utd7.replace('Jackpot', '0')
    
    assert utd7.isdigit(), log_print(f'utdelning 7 rätt är inte ett tal: {utd7}','e')
    
    elem = the_driver.find_element_by_xpath("//span[text()='Utdelning 6 rätt']")
    utd6 = elem.find_element_by_xpath("following-sibling::span").text

    utd6 = utd6.replace(' ', '')
    utd6 = utd6.replace('kr', '')
    utd6 = utd6.replace('Jackpot', '0')
    assert utd6.isdigit(), log_print(f'utdelning 6 rätt är inte ett tal: {utd6}','e')
    
    elem = the_driver.find_element_by_xpath("//span[text()='Utdelning 5 rätt']")
    utd5 = elem.find_element_by_xpath("following-sibling::span").text

    utd5 = utd5.replace(' ', '')
    utd5 = utd5.replace('kr', '')
    utd5 = utd5.replace('Jackpot', '0')
    assert utd5.isdigit(), log_print(f'utdelning 5 rätt är inte ett tal: {utd5}', 'e')
    
    return int(utd7), int(utd6), int(utd5)

# hämta utdelnings-info och spara på fil
def get_utdelning(driver_r, dat, bana):
    utd7, utd6, utd5 = scrape_utdelning(driver_r)
    assert utd7 > utd6 or utd7==0, log_print(f'7 rätt skall ge mer pengar än 6 rätt','e')
    assert utd6 > utd5 or utd6 == 0, log_print(f'6 rätt skall ge mer pengar än 5 rätt', 'e')

    log_print(f'{dat} Utdelning: 7or {utd7}, 6or {utd6}, 5or {utd5}','i')
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
    log_print(f'inkludera_resultat för avd {anr}','i')
    
    res_rader = res_avd.find_elements(By.CLASS_NAME,
            "startlist__row")  # alla rader i loppet
    log_print(f'antal rader i resultat: {len(res_rader)}')
    assert len(res_rader) > 0, log_print(f'Inga resultat hittade för avd {anr}','e')

    res_startnr = res_avd.find_elements(By.CLASS_NAME,
        "horse-u536nn")  # alla startnr
    log_print(f'antal startnr i resultat: {len(res_startnr)}')
    assert len(res_startnr) > 0, log_print(f'Inga startnr hittade för avd {anr}','e')

    d = {'plac': [], 'startnr': []}
    for nr, rad in enumerate(res_rader):
        plac = rad.text.split(' ')[0]

        d['plac'].append(plac)
        d['startnr'].append(res_startnr[nr].text)

    temp_df = pd.DataFrame(d)
    temp_df['startnr'] = temp_df.startnr.astype(int)
    temp_df.sort_values(by='startnr', ascending=True, inplace=True)
    assert len(temp_df) > 0, log_print(f'temp_df tom','e')
    return temp_df.plac.to_list()

# %%

def en_rad(vdict, datum, bana, start, lopp_dist, avd, anr, r, rad, voddss, poddss, strecks, names, pris, history):
    log_print(f'en_rad() {names[r].text} called with avd={anr+1} and vdict["avd"]={vdict["avd"]}')

    vdict['datum'].append(datum)
    vdict['bana'].append(bana)
    vdict['avd'].append(anr+1)
    vdict['startnr'].append(r+1)
    vdict['häst'].append(names[r].text)
    vdict['start'].append(start)
    vdict['lopp_dist'].append(lopp_dist)
    vdict['pris'].append(pris)
    vdict['vodds'].append(voddss[r].text)  # vodds i lopp 1 utan rubrik
    log_print(f'en_rad() r={r} vodds={voddss[r].text}  avd={anr+1}') if anr==0 else None
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
    
    kr = rad.find_elements(By.CLASS_NAME, "earningsPerStart-col")[0].text  # kr/start i lopp 1 utan rubrik
    vdict['kr'].append(kr)

    # dist och spår i lopp 1 utan rubrik
    dist_sp = rad.find_elements(By.CLASS_NAME, "postPositionAndDistance-col")
    
    assert len(dist_sp) > 0, log_print(f'dist_sp ej len>0 len={len(dist_sp)} {dist_sp}','e')
   
    dist_sp_text = dist_sp[0].text.split(':')
    dist = dist_sp_text[0]
    spår = 99
    if len(dist_sp_text) > 1:
        spår = dist_sp_text[1]
    vdict['dist'].append(dist)
    vdict['spår'].append(spår)
    if history:
        log_print(f'history avd {anr+1} för startnr {r+1}') if anr+1==5 else None
        ## history från startlistan ##
        log_print(f'history avd {anr+1} för startnr {r+1}')
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
            log_print(f'len h_dates {len(h_dates)} < 5 avd {anr+1}','w')
            ln=len(h_dates)
            log_print(f'{names[r].text} avd {anr+1} ','i')
            log_print(f'len h_dates {len(h_dates)} avd={anr+1} startnr={r+1}','i')
            log_print(f'len(h_banor) {[len(h_banor)-1]} avd {anr+1}','i')
            log_print(f'datum-text = {dtext} avd={anr+1} för startnr={r+1}','i')
            for h in range(ln, 5):
                fld = 'h'+str(h+1)+'_'
                log_print(f'fld={fld}, h={h}, ln={ln}, avd {anr+1} för startnr {r+1}')
                vdict[fld+'dat'].append(dtext)
                vdict[fld+'bana'].append(h_banor[ln-1].text)
                vdict[fld+'kusk'].append(h_kuskar[ln-1].text)
                vdict[fld+'plac'].append(h_plac[ln-1].text)
                vdict[fld+'dist'].append(h_dist[ln-1].text)
                vdict[fld+'spår'].append(h_spår[ln-1].text)
                vdict[fld+'kmtid'].append(h_kmtid[ln-1].text)
                vdict[fld+'odds'].append(h_odds[ln-1].text)
                vdict[fld+'pris'].append(h_pris[ln-1].text) 
                log_print(f'vdict {vdict}')   
                        
        log_print(f'klart history avd {anr+1} för startnr {r+1}')
    log_print(f'klart rad {r+1} avd {anr+1}')
    return vdict


def do_scraping(driver_s, driver_r, avdelningar, history, datum):  # get data from web site
    
    vdict = {'datum': [], 'bana': [], 'avd': [], 'startnr': [], 'häst': [], 'ålder': [], 'kön': [], 'kusk': [], 'lopp_dist': [],
             'start': [], 'dist': [], 'pris': [], 'spår': [], 'streck': [], 'vodds': [], 'podds': [], 'kr': [], }
    if driver_r:
        vdict.update({'plac': []})
        driver_r.implicitly_wait(10)     # seconds
    if history:
        vdict.update(
            {'h1_dat': [], 'h1_bana': [], 'h1_kusk': [], 'h1_plac': [], 'h1_dist': [], 'h1_spår': [], 'h1_odds': [], 'h1_pris': [], 'h1_kmtid': [],
             'h2_dat': [], 'h2_bana': [], 'h2_kusk': [], 'h2_plac': [], 'h2_dist': [], 'h2_spår': [], 'h2_odds': [], 'h2_pris': [], 'h2_kmtid': [],
             'h3_dat': [], 'h3_bana': [], 'h3_kusk': [], 'h3_plac': [], 'h3_dist': [], 'h3_spår': [], 'h3_odds': [], 'h3_pris': [], 'h3_kmtid': [],
             'h4_dat': [], 'h4_bana': [], 'h4_kusk': [], 'h4_plac': [], 'h4_dist': [], 'h4_spår': [], 'h4_odds': [], 'h4_pris': [], 'h4_kmtid': [],
             'h5_dat': [], 'h5_bana': [], 'h5_kusk': [], 'h5_plac': [], 'h5_dist': [], 'h5_spår': [], 'h5_odds': [], 'h5_pris': [], 'h5_kmtid': [],
             })

    
    if driver_r:
        log_print(f'find driver_r lopp för avd {avdelningar}','i')     
        # söksträng = f"//div[@class='game-view'][starts-with(@name, '{datum}')]"
        söksträng = f"//div[starts-with(@name, '{datum}')]"
        result_tab = driver_r.find_elements_by_xpath(söksträng)

        # result_tab = driver_r.find_elements(
        #     By.CLASS_NAME, 'game-table')[:]  # alla lopp med resultatordning
        log_print(f'2r ############################### len result_tab = {len(result_tab)} avd={avdelningar}', 'i')
        
        if len(result_tab) == 8:
            result_tab = result_tab[1:]

        assert len(result_tab) == 7, log_print(f'################## driver_r  Antal resultat är fel i avd {avdelningar}: {len(result_tab)}','e')
        
    avd = avdelningar[0]    
    driver_s.implicitly_wait(10)     # seconds
    # start_tab = driver_s.find_elements(By.CLASS_NAME, 'game-table')[:]  # alla lopp med startlistor
    # söksträng = f"//div[@class='game-view'][starts-with(@name, '{datum}')]"
    söksträng = f"//div[starts-with(@name, '{datum}')]"
    
    log_print(f'find driver_s lopp för avd {avd} med sök {söksträng}','i')
    start_tab = driver_s.find_elements_by_xpath(söksträng)
    log_print(f'1 ########## driver_s, len start_tab = {len(start_tab)} avd={avdelningar}', 'i')
    if len(start_tab) == 8:
        start_tab = start_tab[1:]

    assert len(start_tab) == 7, log_print(f'############ driver_s Antal lopp är fel: {len(start_tab)} avd {avd}','e')
    
    comb = driver_s.find_elements(By.CLASS_NAME,'race-combined-info')  # alla bana,dist,start
    
    priselement = driver_s.find_elements_by_xpath("//span[starts-with(text(), 'Pris:')]")

    # priselement = driver_s.find_elements(By.CLASS_NAME,
    #     'css-1lnkuf6-startlistraceinfodetails-styles--infoContainer')
    log_print(f'driver_s: len priselement = {len(priselement)} avd={avd}')
    assert len(priselement) == 7, log_print(f'Antal priser är fel: {len(priselement)} avd={avd}','e')
    
    priser = priselement[avd-1].text
    log_print(f'Priser: {priser} avd={avdelningar}', 'i')

    NOK, DKK, EUR = False, False, False
    if 'NOK' in priser:
        NOK = True
    elif 'DKK' in priser:
        DKK = True
    elif 'EUR' in priser:
        EUR = True
    
    # for p in priselement:
    #     log_print(f'priser: p.text {p.text} avd={avdelningar}', 'i')
    #     if 'EUR' in p.text:
    #         EUR = True
    #         break
    #     elif 'NOK' in p.text:
    #         NOK = True
    #         break
    #     elif 'DKK' in p.text:
    #         DKK = True
    #         break
        
    if EUR or NOK or DKK:
        log_print(f'EUR: {EUR}, NOK: {NOK} DKK: {DKK} avd={avd}', 'w')
    else:
        log_print(f'valuta SEK i priser avd={avd}', 'i')    
    # alla lopps priser
    # priser = [p.text for p in priselement if 'Pris:' in p.text]
    
    # if len(priser) == 0:
    #     priser = [p.text for p in priselement 
    #               if 'åriga' not in p.text and 
    #               ('Tillägg' not in p.text) and 
    #               ('EUR' in p.text or 'NOK' in p.text or 'DKK' in p.text)]  # alla lopps priser
    
  
    # ett lopp (de häst-relaterade som inte kan bli 'missing' tar jag direkt på loppnivå)
    for anr, avd_el in enumerate(start_tab):
        log_print(f'for-loop anr={anr} Avd={avd}')
        if anr+1 not in avdelningar:
            continue
        assert anr+1 == avd, log_print(f'Start avd {anr+1} ej samma som avdelningar[0] avd={avd}','e')
        log_print(f'for-loop kör med anr={anr} avd={avd}')
        bana = comb[anr].text.split('\n')[0]
        assert len(bana) > 0, f'Bana saknas: {bana} avd={avd}'
        lopp_dist = comb[anr].text.split('\n')[1].split(' ')[0][:-1]
        assert len(lopp_dist) > 0, f'Loppdist saknas: {lopp_dist} avd={avd}'
        start = comb[anr].text.split('\n')[1].split(' ')[1]
        assert len(start) > 0, f'Start saknas: {start} avd={avd}'
        if EUR:
            pris = priser.split('-')[0] + '0'    # Euro -> SEK
        elif NOK:
            pris = priser.split('-')[0].replace('.', '')  # NOK -> SEK
        elif DKK:
            pris = priser.split('-')[0].replace('.', '')  # DKK -> SEK
        else:
            pris = priser.split('-')[0].split(' ')[1]
            
        pris=pris.replace(' ','')
        pris=pris.replace('.','')
        assert len(pris) > 0, log_print(f'Pris saknas: {pris} avd={avd}','e')
        log_print(f'i loop: pris={pris} Avd={anr+1}')
    
        names = avd_el.find_elements(By.CLASS_NAME, "horse-name")  # alla hästar/kön/åldet i loppet
        assert len(names) > 0, 'no names found avd {avd}'
        voddss = avd_el.find_elements(By.CLASS_NAME, "vOdds-col")[1:]  # vodds i loppet utan rubrik
        log_print(f'len voddss={len(voddss)} avd={avd}')
        assert len(voddss) > 0, 'no vodds found avd {avd}'
        poddss = avd_el.find_elements(By.CLASS_NAME, "pOdds-col")[1:]  # podds i loppet utan rubrik
        assert len(poddss) > 0, 'no podds found avd {avd}'
        
        rader = avd_el.find_elements(By.CLASS_NAME, "startlist__row")  # alla rader i loppet
        log_print(f'len rader={len(rader)} avd={avd}')
        assert len(rader) > 0, 'no rows found   avd {avd}'
        
        # streck i loppet  utan rubrik
        strecks = avd_el.find_elements(By.CLASS_NAME, "betDistribution-col")[1:]
        assert len(strecks) > 0, log_print(f'no strecks found avd {avd}', 'e')
        log_print(f'Antal i avd={avd}: names={len(names)}, vodds={len(voddss)}, podds={len(poddss)}, rader={len(rader)}, streck={len(strecks)}', 'i')
        assert len(names) == len(voddss) == len(poddss) == len(rader) == len(strecks), log_print(f'antal i loppet är fel avd={avdelningar}','e')
        if driver_r:
            log_print(f'driver_r: hämta resultat från resultatsidan avd={avd}', 'i')
            # resultat från resultatsidan
            res_avd = result_tab[anr]
            log_print(f'res_avd={res_avd} avd={avd}')
            assert res_avd, f'no result found: avd={anr+1}'
            # placeringar sorterade efter startnr för en avd
            places = inkludera_resultat(res_avd, anr+1)
            
            log_print(f'places={places} avd={avd}')
            # res_startnr = res_avd.find_elements(By.CLASS_NAME,"css-1jc4209-horseview-styles--startNumber")[1:]
            vdict['plac'] += places    # konkatenera listorna

        # print(f'AVD', anr+1, bana, lopp_dist, start, end=' ')
    
        # from concurrent.futures import ThreadPoolExecutor 
        # from concurrent.futures import as_completed
        
        # vdicts=[]
        # with ThreadPoolExecutor() as e:
        log_print(f'Hämta från alla rader för avd {avd}','i')
        for r, rad in enumerate(rader):
            vdict = en_rad(vdict, datum, bana, start, lopp_dist, avd_el, anr, r, rad, voddss, poddss, strecks, names, pris, history)
     
        log_print(f'Klar avd {anr+1}','i')    
    
    return vdict, bana



# %% popup för vilken info som ska visas
def anpassa(driver_s, avd, datum):

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
        log_print(f'{element_namn}.is_selected={element.is_selected()} avdelning {avd}')

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
    söksträng = f"//div[starts-with(@name, '{datum}')]"
    sl = driver_s.find_elements_by_xpath(söksträng)
    # sl = driver_s.find_elements(
    #     By.CLASS_NAME, "css-8y0fv8-Startlists-styles--startlistelement")  # täcker hela sidan för avd 1-7
    assert len(sl) == 7, log_print(f'len(sl) avd={avd} är inte 7, {len(sl)}','e')
    
    log_print(f'Hitta de tre knapparna "Andra spel", "Utöka alla" och "Anpassa" avd={avd}','i')
    avd_buttons = sl[avd-1].find_elements(
        By.CSS_SELECTOR, "button[class^='MuiButtonBase-root MuiButton-root']")

    assert len(avd_buttons) > 0, log_print(f'avd_buttons skall inte vara tom: {avd_buttons} för avd {avd}','e')

    log_print(f' avd_buttons[0].text  {avd_buttons[0].text}, avd_buttons[1]. text {avd_buttons[1].text}, avd_buttons[2].text {avd_buttons[2].text}')

    # Först utöka alla history för avdelningen
    utöka_button = avd_buttons[1]
    knapp_text = utöka_button.text.replace("\n","")
    log_print(f'Kolla om knappen "Utöka alla" har rätt text {knapp_text}  avd={avd}')
    
    if utöka_button.text.startswith('Utöka'):
        log_print(f' Klickar nu på {knapp_text}, avdelning {avd}')
        actions = ActionChains(driver_s)
        actions.move_to_element(utöka_button).click().perform()

    if utöka_button.text.startswith('Utöka'):  
        log_print(f' Klickar nu med JavaScript på {knapp_text}, avdelning {avd}')
        driver_s.execute_script("arguments[0].click();", utöka_button)

    # Anpassa vilka kolumner i loppet som skall visas
    log_print(f'Välj knappen "Anpassa" avd={avd}')
    anpassa_button = avd_buttons[2]
    knapp_text = anpassa_button.text.replace("\n", "")
    log_print(f' Klickar nu på {knapp_text}, avdelning {avd}', 'i')
    actions = ActionChains(driver_s)
    actions.move_to_element(anpassa_button).click().perform()
    log_print(f' Klickar nu med JavaScript på {knapp_text}, avdelning {avd}', 'i')
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
    log_print(f'Klickade på Aktivera. Färdig med Anpassa-popup för avd={avd}','i')
        

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
    log_print(f'', 'i')
    log_print(f'det tog {round(time.perf_counter() - start_time, 3)} sekunder', 'i')
    
    return df

#%%
def v75_threads(resultat=False, history=False, headless=True, avdelningar=None, driver_s=None, driver_r=None):
    ### Hela loopen med alla lopp i alla veckor i omg_df ###
    log_print(f'avdelningar={avdelningar}')
    omg_df = pd.read_csv(
        'C:\\Users\\peter\\Documents\\MyProjects\\PyProj\\Trav\\spel\\omg_att_spela_link.csv')

    if not driver_r:
        if resultat:
            log_print(f'startar webdriver driver_r för avd={avdelningar}', 'i')
            driver_r = get_webdriver(True,headless)  # get web driver for results
            log_print(f'Startade driver_r för avd={avdelningar}', 'i')
        else:
            driver_r = None

    if not driver_s:
        log_print(f'Startar webdriver driver_s för avd={avdelningar}', 'i')
        driver_s = get_webdriver(False,headless)  # web driver för startlista
        log_print(f'Startade driver_s för avd={avdelningar}', 'i')

    for enum, omg in enumerate(omg_df.Link.values):
        log_print(f'omgång {enum+1}={omg}', 'i')
        datum = omg.split('spel/')[1][0:10]
        df = pd.DataFrame()
        log_print(f'Öppnar startlist-sidan med driver_s för omg={omg} avd={avdelningar}', 'i')
        
        # _=input('Tryck enter för att öppna startlistan\n')
        driver_s.implicitly_wait(10)     # seconds
        driver_s.get(omg)              # öppna startlista
        
        log_print(f'############## Öppnade startlist-sidan med driver_s för avd={avdelningar}')

        if enum == 0:  # första gången startlista
            # ok till första popup om kakor
            
            log_print(f'find cookie popup')
            # _ = input('Tryck enter för find cookie popup\n')
            driver_s.implicitly_wait(20)  # seconds
            but_kakor = driver_s.find_element(By.ID,"onetrust-accept-btn-handler")
            but_kakor_text = but_kakor.text
            
            log_print(f'klickar på {but_kakor_text} för avd={avdelningar}', 'i')
            # _ = input(f'Tryck enter för att klicka på {but_kakor_text}\n')
            
            driver_s.implicitly_wait(10)     # seconds
            but_kakor.click()
            driver_s.fullscreen_window()
            
            log_print(f'klickade på {but_kakor_text} för avd={avdelningar}', 'i')
            
            # _ = input('Tryck enter för att klicka bort reklamen\n')
            
            # try:
            #     driver_s.implicitly_wait(1)     # seconds
            #     driver_s.find_element(By.CLASS_NAME, "css-1m9bhvf-Typography-styles--body1-Typography--Typography").click()
            #     log_print(f'********Så länge reklam för vm i travspel finns kvar gör vi Try hät')
            # except:
            #     print("*********Är VM i travspel bortplockat nu? Ta bort denn kod i så fall.")
            #     pass
            
            # _ =input('Efter Try: Klicka för find lopp-info\n')
            driver_s.implicitly_wait(20)  # seconds
            race_info = driver_s.find_elements(By.CLASS_NAME, "race-info-toggle")
            if len(race_info)==0: #Try again
                log_print(f'try race_info igen för avd {avdelningar}','w')
                input('enter för att försöka igen med race_info')
                # WebDriverWait(driver_s, 10).until(
                #     EC.presence_of_element_located((By.CLASS_NAME, "race-info-toggle")))
                
                driver_s.implicitly_wait(10)  # seconds
                race_info = driver_s.find_elements(By.CLASS_NAME, "race-info-toggle")
                
            log_print(f'len(race_info)= {len(race_info)} för avd {avdelningar} resultat={resultat}')
            assert len(race_info) > 0, log_print(f'race_info måste innehålla en info per lopp men är tom avd={avdelningar} resultat={resultat}', 'e') 
            
            driver_s.find_elements(By.CLASS_NAME, "race-info-toggle")[1].click()  # prissummor mm
            
            # _ = input('Efter prissummor mm: Klicka på OK för att fortsätta\n'  )
            
            log_print(f'Kör "anpassa" med driver_s för avd={avdelningar}')
            anpassa(driver_s,avdelningar, datum)
            log_print(f'Klar "anpassa" med driver_s för avd={avdelningar}')
            # _ = input('Efter anpassa: Klicka på OK för att fortsätta\n'  )
            
        # resultat (driver_r)
        if resultat:
            log_print(f'Öppnar resultatsidan med driver_r för avd={avdelningar}','i')
            driver_r.implicitly_wait(10) # seconds
            
            try:
                driver_r.get(omg+'/resultat')   # öppna resultat
                log_print(f'Öppnade resultatsidan med driver_r för avd {avdelningar}','i')
                if resultat and enum == 0:  # första gången resultat
                    log_print(f'Första gången med driver_r för avd {avdelningar}','i')

                    but_kakor = driver_r.find_element(By.ID, "onetrust-accept-btn-handler")
                    but_kakor.click()
                    driver_r.fullscreen_window()
                    log_print(f'Klickade på {but_kakor.text} med driver_r för avd={avdelningar}', 'i')
                # _ = input('Efter öppnade resultatlistan: Klicka på OK för att fortsätta\n'  )
            except:
                log_print(f'Kunde inte öppna resultatsidan för avd {avdelningar} omgång {enum+1}','w') 
                log_print(f'Sätter resultat=False, driver_r.quit() och driver_r=None \
                            för avd {avdelningar} omgång {enum+1}: {omg}', 'w')
                resultat=False
                driver_r.quit()
                driver_r=None
            
            # Check if warning about 'resultat' is not present
            if resultat:
                warnings = driver_r.find_elements_by_css_selector('.css-qev7py-Alert-styles--StyledAlert-Alert-styles--alertWarning-Alert-styles--StyledAlert[data-test-id="alert-warning"]')
                if len(warnings) > 0:
                    log_print(f"Varningstext: {warnings[0].text} finns på sidan", 'w')
                    log_print(f'Sätter resultat=False, driver_r.quit() och driver_r=None \
                                för avd {avdelningar} omgång {enum+1}: {omg}', 'w')
                    resultat=False
                    driver_r.quit()
                    driver_r=None    
                
        # scraping
        driver_text = 'och driver_r' if driver_r is not None else ''
        log_print(f'Kör scraping med driver_s {driver_text} för avd {avdelningar}','i')
        try:
            komplett, bana = do_scraping(driver_s, driver_r, avdelningar, history, datum)
            log_print(f'Klar scraping med driver_s {driver_text} för avd {avdelningar}','i')
        except:
            log_print(f'************************** Något gick fel i do_scraping avd {avdelningar} - quit drive *******************************','e') 
            quit_drivers(driver_s, driver_r)    
            log_print(f'df.shape={df.shape}','e')
            return df
        
        temp_df = pd.DataFrame(komplett)
        df = pd.concat([df, temp_df])

        # utdelning
        if resultat:
            try:
                if 1 in avdelningar: # Behövs bara en gång
                    log_print(f'Kör utdelning med driver_r för avd={avdelningar}','i')
                    get_utdelning(driver_r, datum, bana)  # utdelning för denna omgång
                    log_print(f'Klar utdelning med driver_r för avd={avdelningar}')
                else:
                    log_print(f'Skippade utdelning med driver_r för avd={avdelningar}','i')
            except:
                log_print(f'************************** Något gick fel i utdelning(...) avd {avdelningar} - quit drivers *******************************','e')
                quit_drivers(driver_s, driver_r)    
                return df
            
    # för att göra tester och debugging efteråt
    # global df_spara
    # df_spara = df.copy()
    ###
    
    quit_drivers(driver_s, driver_r)
    
    # strukna = df.loc[df.vodds == 'EJ', :].copy()  # Ta bort all strukna

    df = städa_och_rensa(df, history)
    
    return df

if __name__ == '__main__':
    """ Här kan vi köra direkt från denna py-fil
    """
    
    logging.basicConfig(level=logging.INFO, filemode='a', filename='v75.log', force=True,
                    encoding='utf-8', format='v75:' '%(asctime)s - %(levelname)s - %(lineno)d - %(message)s')
    logging.info('Startar v75_scraping.py')
    print('Startar v75_scraping.py')
    pd.options.display.max_rows = 200
    import concurrent.futures
    start_time = time.perf_counter()
    log_print(f'START GOING')
    
    ######### settings #########
    omg_df = pd.read_csv(
        'C:\\Users\\peter\\Documents\\MyProjects\\PyProj\\Trav\\spel\\omg_att_spela_link.csv')
    # avd_list=[[1],[2],[3],[4],[5],[6],[7]]
    # avd_list=[[1],[2],[3],[4],[5],[6]]
    # avd_list=[[1],[2],[3],[4],[5]]
    # avd_list=[[1],[2],[3],[4]]
    # avd_list=[[1],[2],[3]]
    # avd_list=[[1],[2]]
    avd_list=[[1]]
    concurrency=False
    resultat=True
    history=True
    headless=False
    log_print(f'Kör med avd_list={avd_list}, resultat={resultat}, history={history}, headless={headless}, omg={omg_df.Link.values}','i')
    
    def start_scrape(avd_list):
        df = v75_threads(resultat=resultat, history=history, headless=headless, avdelningar=avd_list)
        log_print(f'STOP GOING')
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
            
    log_print(f'efter allt klart: df.shape={df.shape}','i')    
    df.sort_values(by=['datum', 'avd', 'startnr', ], inplace=True)        
    # df.to_csv('temp.csv', index=False)
    log_print(f'\ndet tog {round(time.perf_counter() - start_time, 3)} sekunder','i')
    
import datetime    
print('END', datetime.datetime.now().strftime("%H.%M.%S"))

# %%
