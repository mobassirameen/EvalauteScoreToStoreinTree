import os,sys,argparse
import uproot
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import warnings
warnings.filterwarnings('ignore')
import joblib
import multiprocessing as mp
import time

pjoin = os.path.join

def get_inputsamplefiles(inputdir):
    samplelist=os.listdir(inputdir)
    samplelist=[f'{inputdir}/{filename}' for filename in samplelist]
    return samplelist


def return_dataframe_withscore(data,varlist,modelinfo):
    data_nn = data[varlist]
    ##Prepare for evaluation
    X_test, y_test = data_nn.values[:, :-1], data_nn.values[:, -1]
    X_test  = np.asarray(X_test).astype(np.float32)
    y_test  = np.asarray(y_test).astype(np.float32)
    
    n_features = X_test.shape[1]
    print(f'No of Features at evaluation ={n_features}')

    for node in modelinfo:
        modelname =modelinfo[node]["modelname"]
        modelfile =modelinfo[node]["modelfile"]
        scalerfile=modelinfo[node]["scalerfile"]

        mymodel = tf.keras.models.load_model(modelfile)
        mymodel.load_weights(modelfile)
        myscaler=joblib.load(scalerfile)
    
        print(f"Evaluating >>> {modelname}")
        
        X_test_scaled = myscaler.transform(X_test)
        prob          = mymodel.predict(X_test_scaled,batch_size=800000)
        ##Save in dataframe
        data[node]    = prob

        
    data.drop('label',axis=1,inplace=True)
    return data


def evaluate_single_file(infile,branchlist,event_selection,modelinfo,outdir):

    rootfile=uproot.open(infile)
    events  = rootfile["Events"]
    hCount_h= rootfile["hCount"]

    treeName = "Events"
    df = events.arrays(library='pd')
    df['label']=0
    
    #event selection
    df = df.query(event_selection)
    print(">>>-----------------------------------------------")
    print(">>> No of events after selection = ",df.shape[0])
    print(">>>-----------------------------------------------")
    if not(df.shape[0]):
        print("<<<< NOT MAKING TREE. NO EVENTS PASS SELECTION >>>>>")
        
    else:
        #get new dataframe with MVA score
        newdf = return_dataframe_withscore(df,branchlist,modelinfo)
        #newdf['nEvents']=len(metadata['nEvt'].array())
    
        #output root file
        ofname=f"{outdir}/MVA_{infile.split('/')[-1]}"
        file_ = uproot.recreate(ofname)
        file_[treeName]= newdf
        file_['hCount']=hCount_h
        file_.compression = uproot.ZLIB(9)
        
        print()
        print(f'TTree with MVA score is done...')
        print(f"FILENAME >>>{ofname}")
        print(f"NEVENTS  >>>{len(events['lep0_flavor'].array())}")
        print()
        file_.close()
    

def prepare_modelinfo_dict(modelfiledir,VLLmodel,MassPoints,Node,year):
    modelinfo={}
    for Mass in MassPoints:
        for node in Node:
            score_branchname=f"{node}score{VLLmodel}M{Mass}"
            modelinfo[score_branchname]={}
            modelinfo[score_branchname]['modelname']=f"{node}score{VLLmodel}M{Mass}"

            if(VLLmodel=="VLLmu" and Mass==350):Mass=400
            modelinfo[score_branchname]['modelfile']=pjoin(modelfiledir,f"best_model_{node}vs{VLLmodel}M{Mass}_{year}.h5")
            modelinfo[score_branchname]['scalerfile']=pjoin(modelfiledir,f"scaler_{node}vs{VLLmodel}M{Mass}_{year}.save")

    return modelinfo
    
def main(inputdir,outdir,event_selection,multiprocessing=False):
    
    modelfiledir="/home/alaha/work/VLLAnalysis/AnalysisModules/MVA/MVAv2/data/binaryclassifier_data/v2/modeldata/"
    
    Node       = ['wjets','ttjets','qcd','zjets']
    modelinfo_VLLtau  = prepare_modelinfo_dict(modelfiledir,'VLLtau',[100,125,150,200,250,300,350,400],Node,year='Run2')
    modelinfo_VLLmu   = prepare_modelinfo_dict(modelfiledir,'VLLmu',[100,125,150,200,250,300,350,400,450,500,750,1000],Node,year='Run2')
    modelinfo = {**modelinfo_VLLtau, ** modelinfo_VLLmu}

    #print(modelinfo)
    
    branchlist = ['lep0_pt','lep0_eta','lep0_mt','jet0_pt','jet0_eta',
                  'jet0_mt','jet1_pt','jet1_eta', 'jet1_mt','dijet_pt',
                  'dijet_mt','deltaR_jet01','deltaPhi_metjet0',
                  'deltaPhi_metjet1', 'deltaPhi_metlep0','deltaPhi_jet0lep0',
                  'deltaPhi_jet1lep0', 'deltaPhi_dijetlep0','deltaPhi_metdijet',
                  'event_MET', 'event_HT', 'n_Jet','deepjetQG_jet0',
                  'deepjetQG_jet1', 'event_avgQGscore','label']
    

    ##m--------makeTree----------------
    samplefiles=get_inputsamplefiles(inputdir) ##get the filelist
    
    #samplefiles=[
    #    "/data/data_alaha/storage/SEP202024/NtuplesLJJ/2016postVFP/jecdown/SEP202024_Ntuples_1L2J_2016postVFP_HTbinnedWJets_InclusiveStitched_sample.root"
    #]
    
    
    ###-----------    
    if(multiprocessing):
        st=time.time()
        with mp.Pool(processes=mp.cpu_count()) as pool:
            pool.starmap(evaluate_single_file, [(infile, branchlist, event_selection, modelinfo, outdir) for infile in samplefiles])

        en=time.time()
        print(f">>> Multiprocessing ON: total time taken = {en-st:.2f} seconds\n")
    else:
        st=time.time()
        for infile in samplefiles:
            evaluate_single_file(infile,branchlist,event_selection,modelinfo,outdir)
        en=time.time()
        print(f">>> Multiprocessing OFF: total time taken = {en-st:.2f} seconds\n")

def Argument_Parser():
    import argparse
    
    parser=argparse.ArgumentParser()
    parser.add_argument('-i','--inputdir',type=str,required=True ,help='inputdir of sample')
    parser.add_argument('-o','--outdir'  ,type=str,required=True ,default='output/MVAtest/',help='outputdir of sample')
    #parser.add_argument('-y','--year'    ,type=str,required=True ,help='data-taking-era')
    parser.add_argument('-mp','--mproc'  ,type=bool,required=False,default=False,help='multiprocessing feature')
    
    args=parser.parse_args()
    
    return args

if __name__=="__main__":

    #arguments
    args = Argument_Parser()
    
    inputdir= args.inputdir
    #year    = args.year
    outdir  = args.outdir
    mpOpt   = args.mproc
    
    #EVENT SELECTION
    event_selection ="lep0_iso<0.15 & dijet_mass>50 & dijet_mass<110 & deltaR_jet01<2.6 & event_ST>250"
    #event_selection  ="lep0_iso<0.15"
    
    #create output folder if not exist
    print("OUTDIR >>> ",outdir)
    if not os.path.isdir(outdir):os.makedirs(outdir)
    
    #run the code here
    main(inputdir,outdir,event_selection,multiprocessing=mpOpt)
    
