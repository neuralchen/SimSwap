import  json


def readConfig(path):
    with open(path,'r') as cf:
        nodelocaltionstr = cf.read()
        nodelocaltioninf = json.loads(nodelocaltionstr)
        if isinstance(nodelocaltioninf,str):
            nodelocaltioninf = json.loads(nodelocaltioninf)
    return nodelocaltioninf

def writeConfig(path, info):
    with open(path, 'w') as cf:
        configjson  = json.dumps(info, indent=4)
        cf.writelines(configjson)