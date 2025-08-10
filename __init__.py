import os, sys

sys.path.append("/opt/utils/geo_utils/")
try:
    import geoapp.transcribe
    import geoapp.wakeword
    import cogs.lookup    
    import cogs.imgreco    
except Exception as e:
    print(f"\n*******\n***** ERROR LOADING\n{e}\n*******\n*******")
    pass


