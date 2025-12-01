import abfile
import os
import shutil
import numpy as np
import argparse

def assemble(date, mem):
    vars_list = ['temp', 'saln', 'dp', 'u', 'v', 'ECO_fla', 'ECO_dia', 'ECO_ccl', 'ECO_flac', 'ECO_diac', 'ECO_cclc']
    layers_list = list(range(1, 51))

    Template_restart_file = '/cluster/work/users/antber/TP2a0.10/expt_02.5/LR_upsampled/restart.' + str(date) + '_00_0000_mem' + mem + '.a'
    Template_restart = abfile.ABFileRestart(Template_restart_file,"r",idm=800,jdm=760)    

    ## To copy the first 10 dp from a HR restart
    Template_HR_file = '/cluster/work/users/antber/dummy/restartTP5/restart.2019_267_00_0000.a'
    Template_HR = abfile.ABFileRestart(Template_HR_file,'r',idm=800,jdm=760)

    Assembled_restart_file = '/cluster/work/users/antber/TP2a0.10/expt_02.5/SR_fields/restart.' + str(date) + '_00_0000_mem' + mem + '.a'
    Assembled_restart = abfile.ABFileRestart(Assembled_restart_file,"w",idm=800,jdm=760)
    Assembled_restart.write_header(25, Template_restart._iversn, Template_restart._yrflag, Template_restart._sigver, Template_restart._nstep, Template_restart._dtime, Template_restart._thbase)

    for keys in sorted( Template_restart.fields.keys() ):
            fieldname = Template_restart.fields[keys]["field"]
            k         = Template_restart.fields[keys]["k"]
            t         = Template_restart.fields[keys]["tlevel"]
            if fieldname in vars_list and k in layers_list:
                if fieldname == 'dp' and k < 10:
                    field = Template_HR.read_field(fieldname,k,t)
                    Assembled_restart.write_field(field,True,fieldname,k,t)
                else:
                    Part_restart_file = '/cluster/work/users/antber/TP2a0.10/expt_02.5/SR_fields/restart.' + str(date) + '_00_0000_mem' + mem + f'_part_{fieldname}_{k}.a'
                    Part_restart = abfile.ABFileRestart(Part_restart_file,"r",idm=800,jdm=760)
                    SRfield     = Part_restart.read_field(fieldname,k,1)
                    Part_restart.close()
                    Assembled_restart.write_field(SRfield,True,fieldname,k,t)
            else:
                field = Template_restart.read_field(fieldname,k,t)
                Assembled_restart.write_field(field,True,fieldname,k,t)

    Template_restart.close()
    Assembled_restart.close()

if __name__ == "__main__":
    # Define the argument parser
    parser = argparse.ArgumentParser(description="Assemble restarts")    
    parser.add_argument('date', type=str, help='The date in format yyyy_ddd, for instance 2019_273.')     
    parser.add_argument('mem', type=int, help='The member number.')

    # Parse the command-line arguments 
    args = parser.parse_args()
    formatted_mem = f"{args.mem:03}"
    assemble(args.date, formatted_mem)
