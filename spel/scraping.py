# -*- coding: utf-8 -*-

# %%


def getBana(the_driver, avd, after=False):
    hack = '1' if avd == '7' else '1'    # när jag kör innan

    #print('bana avd', avd, 'after', after, 'hack', hack)
    end_string = '/div['+hack+']/div/div[1]/div/div[2]/div[2]/div[1]/span'
    if after:
        hack = '2' if avd == '7' else '1'     # när jag kör direkt efter omgångens slut
        end_string = '/div['+hack + \
            ']/div[1]/div[1]/div/div[2]/div[2]/div[1]/span'

    pth = '//*[@id="main"]/div[3]/div[2]/div/div/div/div/div/div[7]/div[' + \
        avd+']/div/div' + end_string

    bana = the_driver.find_element_by_xpath(pth)

    return bana.text

# %%


def getLoppDist(the_driver, avd, after=False):
    hack = '1' if avd == '7' else '1'  # när jag kör innan

    end_string = '/div['+hack+']/div/div/div/div[2]/div[2]/div[2]/span'
    if after:
        hack = '2' if avd == '7' else '1'     # när jag kör direkt efter omgångens slut
        end_string = '/div['+hack + \
            ']/div[1]/div[1]/div/div[2]/div[2]/div[2]/span'

    pth = '//*[@id="main"]/div[3]/div[2]/div/div/div/div/div/div[7]/div[' + \
        avd + ']/div/div' + end_string

    d = the_driver.find_element_by_xpath(pth)
    # lopp distans (loppets distans, hästens kan skilja)
    lopp_dist = d.text.split()[0]
    start = d.text.split()[1]

    return lopp_dist, start

# %%


def getNamn(the_driver, avd, i, after=False):

    end_string = '/td[1]/span[2]/div/span[1]'
    if after:
        end_string = '/td[1]/span/div/span[1]'

    pth = '//*[@id="main"]/div[3]/div[2]/div/div/div/div/div/div[7]/div[' + \
        avd + ']/div/div/table/tbody/tr[' + str(i) + ']' + end_string

    namn = the_driver.find_element_by_xpath(pth)
    return namn.text

# %%


def getKön(the_driver, avd, i, after=False):

    end_string = '/td[1]/span[2]/div/span[2]'
    if after:
        end_string = '/td[1]/span/div/span[2]'

    pth = '//*[@id="main"]/div[3]/div[2]/div/div/div/div/div/div[7]/div[' + \
        avd+']/div/div/table/tbody/tr['+str(i)+']' + end_string

    try:
        kön = the_driver.find_element_by_xpath(pth)
        return kön.text
    except:
        print('inget kön, ingen ålder')
        # logging.warning('både kön och ålder saknas för häst '+str(i))
        return ' '

# %%


def getÅlder(the_driver, avd, i, after=False):

    end_string = '/td[1]/span[2]/div/span[3]'
    if after:
        end_string = '/td[1]/span/div/span[3]'

    pth = '//*[@id="main"]/div[3]/div[2]/div/div/div/div/div/div[7]/div[' + \
        avd+']/div/div/table/tbody/tr['+str(i)+']' + end_string

    try:
        ålder = the_driver.find_elements_by_xpath(pth)
    except:
        # logging.warning('ålder eller kön saknas för häst '+str(i))
        return 0

    if len(ålder) == 0:
        # logging.warning('len(ålder)==0. Ålder eller kön saknas för häst '+str(i))
        return 0

    return ålder[0].text


# %%


def getSpår(the_driver, avd, i, after=False):  # spår och distans
    # sp = '7'  # blev fel direkt efter omgången
    sp = '5'  # gamla vodds

    pth = '//*[@id="main"]/div[3]/div[2]/div/div/div/div/div/div[7]/div[' + \
        avd+']/div/div/table/tbody/tr['+str(i)+']/td['+sp+']'

    sp_dist = the_driver.find_element_by_xpath(pth)
    # print(sp.text)
    txt = sp_dist.text.split()
    spår = None
    dist = None

    if len(txt) > 0:
        dist = txt[0]

    if len(txt) == 3:
        spår = txt[2]

    return spår, dist

# %%


def getKusk(the_driver, avd, i, after=False):

    pth = '//*[@id="main"]/div[3]/div[2]/div/div/div/div/div/div[7]/div[' + \
        avd+']/div/div/table/tbody/tr['+str(i)+']/td[2]/span'
    kusk = the_driver.find_elements_by_xpath(pth)

    return kusk[0].text

# %%


def getStreck(the_driver, avd, i, after=False):

    pth = '//*[@id="main"]/div[3]/div[2]/div/div/div/div/div/div[7]/div[' + \
        avd+']/div/div/table/tbody/tr['+str(i)+']/td[3]'
    streck = the_driver.find_element_by_xpath(pth)
    return streck.text

# %%


def getVodds(the_driver, avd, i, after=False):
    # vo = '7'  #blev fel direkt efter omg
    vo = '6'  #

    pth = '//*[@id="main"]/div[3]/div[2]/div/div/div/div/div/div[7]/div[' + \
        avd+']/div/div/table/tbody/tr['+str(i)+']/td['+vo+']'

    vodds = the_driver.find_element_by_xpath(pth)

    return vodds.text

# %%


def getPodds(the_driver, avd, i,  after=False):
    # po = '6'  # blev fel  direkt efter omg
    po = '7'  #

    pth = '//*[@id="main"]/div[3]/div[2]/div/div/div/div/div/div[7]/div[' + \
        avd+']/div/div/table/tbody/tr['+str(i)+']/td['+po+']'

    podds = the_driver.find_element_by_xpath(pth)

    # print('getPodds', podds.text)

    return podds.text

# %%


def getKrPerStart(the_driver, avd, i,  after=False):
    kr = '4'
    pth = '//*[@id="main"]/div[3]/div[2]/div/div/div/div/div/div[7]/div[' + \
        avd+']/div/div/table/tbody/tr['+str(i)+']/td['+kr+']'
    kronor = the_driver.find_element_by_xpath(pth)

    # kr = driver_s.find_element_by_xpath(pth).text
    return kronor.text
