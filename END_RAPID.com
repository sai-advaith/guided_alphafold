#! /bin/tcsh -f
#
#	"master" script for calculating an absolute Electron Number Density (END) map and
#	a Refinement Against Perturbed Input Data (RAPID) map of the expected noise
#	 based on a phenix.refine input *.eff file.	
#
#									-James Holton 4-14-15
#
#

set eff_file = ""
set prefix     = find_F000
set phenix_log = find_F000_001.log
set cycles     = 5
set RAPID_itrs = 5
set CPUs       = auto

set path = ( `dirname $0` $path )
set tempfile = ${CCP4_SCR}/F000_temp$$


set reso    = ""
set ksol    = "auto"
set Bsol    = "auto"
set Rshrink = "auto"
set Rsolv   = "auto"

foreach arg ( $* )

    if("$arg" =~ *.eff) then
	if(-e "$arg") then
	    set eff_file = "$arg"
	else
	    echo "WARNING: $arg does not exist."
	endif
    endif
    if("$arg" =~ -noscale* ) set NO_SCALE
    if("$arg" =~ -norapid* ) set NO_RAPID
    if("$arg" =~ -nosig* ) set NO_SIGF_RAPID
    if("$arg" =~ -nofofc* ) set NO_FOFC_RAPID
    if("$arg" =~ -nohydro* ) set NO_HADD
#    if("$arg" =~ -skipend* ) set NO_END_MAP

    if("$arg" =~ *A) set reso = `echo $arg | awk '{print $1+0}'`
    if("$arg" =~ cpus=*) set CPUs = `echo $arg | awk -F "=" '{print $2+0}'`
    if("$arg" =~ seeds=*) set RAPID_itrs = `echo $arg | awk -F "=" '{print $2+0}'`
    if("$arg" =~ cycles=*) set cycles = `echo $arg | awk -F "=" '{print $2+0}'`
    if("$arg" =~ ksol=*) set ksol = `echo $arg | awk -F "=" '{print $2+0}'`
    if("$arg" =~ Bsol=*) set Bsol = `echo $arg | awk -F "=" '{print $2+0}'`
    if("$arg" =~ Rsol*=*) set Rsol = `echo $arg | awk -F "=" '{print $2+0}'`
    if("$arg" =~ Rshr*=*) set Rshrink = `echo $arg | awk -F "=" '{print $2+0}'`

end
if(! -e "$eff_file") then
    set BAD = "cannot find phenix.refine eff file"
    goto exit
endif

if("$CPUs" == "auto") then
    if(-e /proc/cpuinfo) set CPUs = `grep processor /proc/cpuinfo |& wc -l`
    if("$CPUs" == "") set CPUs = 1

    set rounds = `echo $RAPID_itrs $CPUs | awk '{print int($1/$2+0.99999)}'`
    set cpu_per_round = `echo $RAPID_itrs $rounds | awk '{print int($1/$2)+1}'`
    set needed_rounds = `echo $RAPID_itrs $cpu_per_round | awk '{for(j=$1;j>0;j-=$2){++n} ; print n}'`
    set needed_CPUs   = `echo $RAPID_itrs $needed_rounds | awk '{print int($1/$2+0.999999)}'`

    echo "$CPUs CPUs are available, so we need at least $needed_rounds rounds to do $RAPID_itrs seeds.  $needed_CPUs CPUs at a time will do."
    set CPUs = "$needed_CPUs"
endif


# edit the eff file to make sure these are volume-scale maps
cat $eff_file |\
awk '{tag[brace+0]=$1}\
     /[{]/{++brace} /[}]/{--brace}\
   tag[0]=="refinement" && tag[1]=="electron_density_maps" && /scale/ && /sigma/{\
	$0="scale=volume"}\
   tag[0]=="refinement" && tag[1]=="electron_density_maps" && /apply/ && /scaling/{\
	$0="apply_volume_scaling=True\napply_sigma_scaling=False"}\
  $1=="serial"{$NF=1;print "serial_format = \"%03d\""}\
  /%d/{gsub("%d","%03d")}\
   {print}' |\
cat >! find_F000_input.eff

setenv PHENIX_OVERWRITE_ALL true

#if($?NO_END_MAP) goto 


# report
echo "will use $CPUs CPUs to do $RAPID_itrs seeds of $cycles macro-cycles each"


if(! $?DEBUG) then
echo "running initial phenix.refine job..."
phenix.refine find_F000_input.eff output.prefix=find_F000 \
  main.number_of_macro_cycles=$cycles \
  export_final_f_model=True  >&! $phenix_log
if($status) then
    cat $phenix_log
    set BAD = "initial phenix.refine job failed."
    goto exit
endif
endif

set pdbfile = ${prefix}_001.pdb
set mtzfile = ${prefix}_001.mtz 
set efffile = ${prefix}_001.eff

# try other names...
if(! -s "$mtzfile" && -e ${prefix}_001_map_coeffs.mtz) cp -p ${prefix}_001_map_coeffs.mtz $mtzfile
if(! -s "$mtzfile" && -e ${prefix}_1_map_coeffs.mtz) cp -p ${prefix}_1_map_coeffs.mtz $mtzfile
if(! -s "$mtzfile" && -e ${prefix}_001.mtz) cp -p ${prefix}_001.mtz $mtzfile
if(! -s "$mtzfile" && -e ${prefix}_1.mtz) cp -p ${prefix}_1.mtz $mtzfile
if(! -s "$pdbfile" && -e ${prefix}_1.pdb) cp -p ${prefix}_1.pdb $pdbfile
if(! -s "$efffile" && -e ${prefix}_1.eff) cp -p ${prefix}_1.eff $efffile

# check that it did not crash?
if(! -s "$pdbfile") then
    set BAD = "pdb file: $pdbfile does not exist."
    goto exit
endif
if(! -s "${mtzfile}") then
    set BAD = "mtz file: ${mtzfile} does not exist."
    goto exit
endif
if(! -s "${efffile}") then
    set BAD = "eff file: ${efffile} does not exist."
    goto exit
endif


# keep all phenix options except the mtz and pdb file names
cat ${efffile} |\
awk '{tag[brace+0]=$1}\
     /[{]/{++brace} /[}]/{--brace}\
   tag[0]=="refinement" && tag[1]=="input" && tag[2]=="pdb"{next}\
   tag[0]=="refinement" && tag[1]=="input" && tag[2]=="xray_data"{next}\
   {print}' |\
cat >! ${prefix}.eff



# now make the "default" 2FoFc map - not neccesarily right scale
echo "calculating traditional 2FoFc.map"
fft hklin ${mtzfile} mapout ${tempfile}ffted.map << EOF > /dev/null
labin F1=2FOFCWT PHI=PH2FOFCWT
EOF
echo "xyzlim asu" |\
 mapmask mapin ${tempfile}ffted.map mapout 2FoFc.map > /dev/null
rm -f ${tempfile}ffted.map

# make sure we use the same grid spacing from here on out
echo "" | mapdump mapin 2FoFc.map >! ${tempfile}mapdump.txt
set xyzgrid = `awk '/Grid sampling on x, y, z ../{print $8,$9,$10}' ${tempfile}mapdump.txt  | tail -1`
echo "using map grid: $xyzgrid"
rm -f ${tempfile}mapdump.txt


# now do the fofc map
fft hklin ${mtzfile} mapout ${tempfile}ffted.map << EOF > /dev/null
labin F1=FOFCWT PHI=PHFOFCWT
GRID $xyzgrid
EOF
echo "xyzlim asu" |\
 mapmask mapin ${tempfile}ffted.map mapout FoFc.map > /dev/null
rm -f ${tempfile}ffted.map







# figure out space group from PDB file
set CELL = `awk '/^CRYST1/{print $2,$3,$4,$5,$6,$7;exit}' $pdbfile`
if( $#CELL != 6) then
    set BAD = "no cell in PDB"
    goto exit
endif
set pdbSG = `awk '/^CRYST1/{SG=substr($0,56,14);if(length(SG)==14)while(gsub(/[^ ]$/,"",SG));print SG;exit}' $pdbfile | head -1`
if("$pdbSG" == "R 32") set pdbSG = "R 3 2"
#if("$pdbSG" == "P 21") set pdbSG = "P 1 21 1"
#if("$pdbSG" == "A 2") set pdbSG = "A 1 2 1"
if("$pdbSG" == "I 21") set pdbSG = "I 1 21 1"
if("$pdbSG" == "A 1") set pdbSG = "P 1"
if("$pdbSG" == "P 21 21 2 A") set pdbSG = "P 21 21 2"
if("$pdbSG" == "P 1-") set pdbSG = "P -1"
if("$pdbSG" == "R 3 2" && $CELL[6] == 120.00) set pdbSG = "H 3 2"
if("$pdbSG" == "R 3" && $CELL[6] == 120.00) set pdbSG = "H 3"
set SGnum = `awk -v pdbSG="$pdbSG" -F "[\047]" 'pdbSG==$2 || pdbSG==$4{print;exit}' ${CLIBD}/symop.lib | awk '{print $1}'`
set SG = `awk -v SGnum=$SGnum 'SGnum==$1{print $4}' ${CLIBD}/symop.lib`
set symops = `awk -v SGnum=$SGnum 'SGnum==$1{print $2}' ${CLIBD}/symop.lib`
if("$SG" == "") set SG = "$pdbSG"

if("$symops" == "") then
    set BAD = "unable to determine number of symmetry operators"
    goto exit
endif


# need cell volume to compute F000 from map offsets
echo $CELL |\
awk 'NF==6{DTR=atan2(1,1)/45; A=cos(DTR*$4); B=cos(DTR*$5); G=cos(DTR*$6); \
 skew = 1 + 2*A*B*G - A*A - B*B - G*G ; if(skew < 0) skew = -skew;\
 printf "%.3f\n", $1*$2*$3*sqrt(skew)}' |\
cat >! ${tempfile}volume
set CELLvolume = `cat ${tempfile}volume`
rm -f ${tempfile}volume
rm -f ${tempfile}maphead.txt

if("$CELLvolume" == "") then
    set BAD = "unable to determine unit cell volume"
    goto exit
endif





# get the optimized ksol and Bsol out of the phenix.refine stdout log
# this is for phenix 1.6.x
if("$ksol" == "auto" || "$Bsol" == "auto") then

  set test = `awk '/k_sol/,/----/{print $3}' $pdbfile | awk 'NF>0' | tail -1`
  if("$test" != "" && "$ksol" == "auto") then
    set ksol = "$test"
    echo "got ksol = $ksol from $pdbfile"
  endif
  set test = `awk '/k_sol/,/----/{print $4}' $pdbfile | awk 'NF>0' | tail -1`
  if("$test" != "" && "$Bsol" == "auto") then
    set Bsol = "$test"
    echo "got Bsol = $Bsol from $pdbfile"
  endif

 if(-e "$phenix_log") then
  set test = `awk '/k_sol/,/----/{print $2}' $phenix_log | awk 'NF>0' | tail -1`
  if("$test" != "" && "$ksol" == "auto") then
    set ksol = "$test"
    echo "got ksol = $ksol from $phenix_log"
  endif
  set test = `awk '/k_sol/,/----/{print $3}' $phenix_log | awk 'NF>0' | tail -1`
  if("$test" != "" && "$Bsol" == "auto") then
    set Bsol = "$test"
    echo "got Bsol = $Bsol from $phenix_log"
  endif
 endif

endif


# this is for phenix 1.8.x
if("$ksol" == "auto" || "$Bsol" == "auto") then

 if(-e "$phenix_log") then
  set test = `awk '/kmask/{getline;print $NF}' $phenix_log | awk 'NF>0' | tail -1`
  if("$test" != "" && "$ksol" == "auto") then
    set ksol = "$test"
    echo "got ksol = kmask = $ksol from $phenix_log"
    set Bsol = 0
    echo "assuming Bsol = $Bsol (does not really matter)"
  endif
 endif

endif
if("$ksol" == "auto") then
    set BAD = "unable to obtain k_sol.  Perhaps supply on command line? "
    goto exit
endif
# Bsol does not change results
if("$Bsol" == "auto") set Bsol = 0

set phenix_reso = `awk '/RESOLUTION RANGE HIGH/{print $NF*0.98}' $pdbfile | tail -1`
set finereso = `echo $phenix_reso | awk '{print $1/2}'`
set reso = $phenix_reso
#if("$reso" == "") set reso = "$phenix_reso"


# assume Pavel took the lowest R?
tac $phenix_log |&\
awk -F "=" '/r_solv/{++p} p{print $NF,$0} p && NF==0{exit}' |\
sort -g | awk '/r_solv/{print $3,$5,$1; exit}' >! ${tempfile}params.txt
set test = `awk '{print $1}' ${tempfile}params.txt`
if($test != "" && "$Rsolv" == "auto") then
    set Rsolv = "$test"
    echo "got Rsolv= $Rsolv from $phenix_log"
endif
set test = `awk '{print $2}' ${tempfile}params.txt`
if($test != "" && "$Rshrink" == "auto") then
    set Rshrink = "$test"
    echo "got Rshrink= $Rshrink from $phenix_log"
endif
rm -f ${tempfile}params.txt

set file = `ls -1t | awk '/.eff$/{print;exit}'`
if(-e "$file") then

    tac "$file" |&\
    awk -F "=" '/shrink_truncation_radius/{print $NF; exit}' |\
    cat >! ${tempfile}params.txt
    set test = `awk '{print $1}' ${tempfile}params.txt`
    if($test != "" && "$Rshrink" == "auto") then
        set Rshrink = "$test"
        echo "got Rshrink= $Rshrink from $file"
    endif

    tac "$file" |&\
    awk -F "=" '/solvent_radius/{print $NF; exit}' |\
    cat >! ${tempfile}params.txt
    set test = `awk '{print $1}' ${tempfile}params.txt`
    if($test != "" && "$Rsolv" == "auto") then
        set Rsolv = "$test"
        echo "got Rsolv= $Rsolv from $file"
    endif
endif
rm -f ${tempfile}params.txt





echo "bulk solvent paramters: $ksol $Bsol $Rsolv $Rshrink"








# set all B factors to 80
echo "setting all B factors to 80 (does not change F000)"
cat $pdbfile |\
awk '/^ATOM/ || /^HETAT/{print substr($0,1,60)" 80.00"substr($0,67);next} ! /^ANISO/{print}' |\
cat >! refined.pdb

# check if hydrogens have already been added
set gotH = `awk '/^ATOM/{++count[$NF]} END{if(count["C"]) print (count["H"]/count["C"] > 0.5)}' $pdbfile`
if("$gotH" != "1" && ! $?NO_HADD) then
    echo "Adding hydrogens..."
    phenix.ready_set refined.pdb
    if(! -s refined.updated.pdb) then
	set BAD = "unable to account for hydrogens"
	goto exit
    endif
    mv refined.updated.pdb refined.pdb
endif

# count up atom types - weigthed by occupancy
cat refined.pdb |\
awk '/^ATOM|^HETAT/{count[$NF]+=substr($0, 55, 6);} \
  END{for(Ee in count) print count[Ee],Ee}' |\
cat >! ${tempfile}Ee_counts.txt

# get a list of atomic numbers
cat ${CLIBD}/atomsf.lib |\
awk '/^[A-Z]/{Ee=$1;getline;print Ee,$2}' |\
cat - ${tempfile}Ee_counts.txt |\
awk '/^[A-Z]/{Z[$1]=$2;next}\
   {Zsum+=$1*Z[$2]} END{print Zsum}' |\
cat >! ${tempfile}Zsum.txt
set Zsum = `cat ${tempfile}Zsum.txt`
set Zsum_F000 = `echo $Zsum $symops | awk '{print $1*$2}'`
set Zsum_offset = `echo $Zsum_F000 $CELLvolume | awk '{print $1/$2}'`


# convert to dummy atoms for SFALL (electron count only)
cat refined.pdb |\
awk '/^CRYST/{print}\
      /^ATOM|^HETAT/{\
        X = substr($0, 31, 8);\
        Y = substr($0, 39, 8);\
        Z = substr($0, 47, 8);\
        O = substr($0, 55, 6);\
       Ee = toupper($NF);\
    printf("ATOM %6d %2s   DUM     1    %8.3f%8.3f%8.3f%5.2f 80.00%12s\n",++n,substr(Ee,1,2),X,Y,Z,O,Ee);}\
    END{print "END"}' |\
  cat >! sfallme.pdb

set Ees = `awk '/^ATOM/{print $NF}' sfallme.pdb | sort -u`

# generate phenix-derived model Fs (but without F000)
echo "generating phenix models..."

# now with bulk solvent turned off
echo "no solvent"
rm -f nobulk.mtz
phenix.fmodel refined.pdb k_sol=0 b_sol=0 high_res=$reso \
  file_name=nobulk.mtz >! phenix_fmodel.log
if(! -s nobulk.mtz) then
    set BAD = "unabel to run phenix.fmodel"
    goto exit
endif

# sharp solvent mask (should have same Q)
echo "sharp solvent"
rm -f ss.mtz
set extra = ""
if("$Rsolv" != "") set extra = "mask.solvent_radius=$Rsolv"
if("$Rshrink" != "") set extra = ( $extra mask.shrink_truncation_radius=$Rshrink )
phenix.fmodel refined.pdb k_sol=$ksol b_sol=0 high_res=$reso \
  file_name=ss.mtz $extra >! phenix_fmodel.log
if(! -s ss.mtz) then
    set BAD = "unabel to run phenix.fmodel"
    goto exit
endif


echo "regular"
rm -f fmodel.mtz
phenix.fmodel $pdbfile k_sol=$ksol b_sol=$Bsol high_res=$phenix_reso \
  file_name=fmodel.mtz $extra >! phenix_fmodel.log
if(! -s fmodel.mtz) then
    set BAD = "unabel to run phenix.fmodel"
    goto exit
endif



# compare with sfall (no anisotropic Bs! )
echo "making sfall map for vacuum level reference"
sfall xyzin sfallme.pdb mapout ${tempfile}sfall.map << EOF >! sfall.log
mode atmmap
SYMM $SG
resolution $reso 1000
FORMFACTOR ngauss 5 $Ees
EOF
echo "xyzlim ASU" | mapmask mapin ${tempfile}sfall.map mapout sfall.map > /dev/null

if(! -s sfall.map) then
    set BAD = "unabel to run sfall"
    goto exit
endif


# examine this map to get the grid and axis convention for this SG
echo "" | mapdump mapin sfall.map >! ${tempfile}maphead.txt
set sfall_mean = `awk '/Mean density/{print $NF}' ${tempfile}maphead.txt`
set model_vac = `echo $sfall_mean | awk '{print -$1}'`
set model_sig = 0
set atom_F000 = `echo $CELLvolume $sfall_mean 0 | awk '{print $1*($2-$3)}'`
echo "no-solvent mean=0 map vacuum level: $model_vac"
#echo "no-solvent F000 = $atom_F000"

echo "Z-sum predicted no-solvent vacuum level: -$Zsum_offset"









# make the maps
echo "calculating maps of phenix models..."


fft hklin ss.mtz mapout ${tempfile}ffted.map << EOF > /dev/null
labin F1=FMODEL PHI=PHIFMODEL
GRID $xyzgrid
EOF
echo "xyzlim asu" |\
 mapmask mapin ${tempfile}ffted.map mapout fmodel_ss.map > /dev/null
rm -f ${tempfile}ffted.map


fft hklin nobulk.mtz mapout ${tempfile}ffted.map << EOF > /dev/null
labin F1=FMODEL PHI=PHIFMODEL
GRID $xyzgrid
EOF
echo "xyzlim asu" |\
 mapmask mapin ${tempfile}ffted.map mapout nobulk.map > /dev/null
rm -f ${tempfile}ffted.map





# check if we need map_vacuum_level.com
which map_vacuum_level.com >& /dev/null
if($status) then
    goto deploy_map_vacuum_level
endif
got_map_vacuum_level:



# subtract to isolate solvent region
echo "subtracting nobulk map from sharp-solvent map to obtain solvent-only map"
echo scale factor -1 | mapmask mapin nobulk.map mapout neg.map > /dev/null
echo "maps add" | mapmask mapin1 fmodel_ss.map mapin2 neg.map mapout sharpsolvent.map > /dev/null

# this is the median difference between solvent on and off (proportional to total charge)
echo "finding lower median with outlier rejection"
if(! $?DEBUG) then
    map_vacuum_level.com sharpsolvent.map >! sharpsolvent_vac.log 
endif
grep "vacuum level:" sharpsolvent_vac.log
set solvent_vac = `awk '/vacuum level:/{print $4}' sharpsolvent_vac.log`
set solvent_sig = `awk '/vacuum level:/{print $6}' sharpsolvent_vac.log`
if(-e hist0.txt ) cp hist0.txt sharpsolvent_hist0.txt

if("$solvent_vac" == "") then
    set BAD = "map_vacuum_level.com failed.  Is it missing? "
    goto exit
endif




# this should be the vacuum level of the 2FoFc (and fmodel) maps
set baseline = `echo $model_vac $solvent_vac | awk '{print -$1-$2}'`
set baseline_sig = `echo $model_sig  $solvent_sig | awk '{print sqrt($1*$1+$2*$2)}'`

#echo scale factor 1 $baseline |\
# mapmask mapin fmodel.map mapout fmodel_vac0.map > /dev/null
#echo "" | mapdump mapin fmodel_vac0.map | grep dens >! dens.txt
#set mean = `awk '/Mean density/{print $NF}' dens.txt`
#set F000 = `echo $CELLvolume $mean 0 | awk '{print $1*($2-$3)}'`
set SIGF000 = `echo $CELLvolume $baseline_sig | awk '{print $1*$2}'`
#echo "fmodel_vac0.map should have a vacuum level of zero and F000 = $F000 +/- $SIGF000"


echo "putting map coefficients on absolute scale"
cad hklin1 fmodel.mtz hklin2 ${mtzfile} hklout cadded.mtz << EOF > /dev/null
labin file 1 E1=FMODEL
labou file 1 E1=Frightscale
labin file 2 E1=2FOFCWT
labou file 2 E1=Fmap
EOF
scaleit hklin cadded.mtz hklout scaled.mtz << EOF >! scaleit_map.log
refine scale
labin FP=Frightscale SIGFP=Fmap FPH1=Fmap SIGFPH1=Fmap
EOF
#egrep "The equivalent isotropic temperature factor is|Derivative   1  |TOTALS" scaleit_map.log
set map_scale = `awk '/Derivative   1  /{print $3+0}' scaleit_map.log | tail -1`
echo "scale factor: $map_scale"
rm -f  ${tempfile}scale.txt
if($?NO_SCALE) then
    echo "but no scaling will be applied."
    set map_scale = 1
endif

echo "applying $map_scale to map coefficients"
cad hklin1 ${mtzfile} hklout scaled_map_coeffs.mtz << EOF > /dev/null
labin file 1 all
scale file 1 $map_scale 0
EOF


# now make the "scaled" 2FoFc map
echo "calculating 2FoFc_scaled.map on absolute scale"
fft hklin scaled_map_coeffs.mtz mapout ${tempfile}ffted.map << EOF > /dev/null
labin F1=2FOFCWT PHI=PH2FOFCWT
GRID $xyzgrid
EOF
echo "xyzlim asu" |\
 mapmask mapin ${tempfile}ffted.map mapout 2FoFc_scaled.map > /dev/null
rm -f ${tempfile}ffted.map


# now do the fofc map
echo "calculating FoFc_scaled.map on absolute scale"
fft hklin scaled_map_coeffs.mtz mapout ${tempfile}ffted.map << EOF > /dev/null
labin F1=FOFCWT PHI=PHFOFCWT
GRID $xyzgrid
EOF
echo "xyzlim asu" |\
 mapmask mapin ${tempfile}ffted.map mapout FoFc_scaled.map > /dev/null
rm -f ${tempfile}ffted.map


#echo scale factor $map_scale 0 |\
# mapmask mapin 2FoFc.map mapout 2FoFc_scaled.map > /dev/null
#echo "" | mapdump mapin 2FoFc_scaled.map | grep dens >! dens.txt
#echo "2FoFc_scaled.map is on the absolute scale"



echo "adding $baseline to 2FoFc_scaled.map to form 2FoFc_END.map"
echo scale factor 1 $baseline |\
 mapmask mapin 2FoFc_scaled.map mapout 2FoFc_END.map > /dev/null
echo "" | mapdump mapin 2FoFc_END.map | grep dens >! dens.txt
set mean = `awk '/Mean density/{print $NF}' dens.txt`
set F000 = `echo $CELLvolume $mean 0 | awk '{print $1*($2-$3)}'`


echo "placing original data on absolute scale in kickme.mtz"
set fmodelfile = ${prefix}_001_f_model.mtz
if(! -e $fmodelfile && -e ${prefix}_1_f_model.mtz) set fmodelfile = ${prefix}_1_f_model.mtz
cad hklin1 fmodel.mtz hklin2 $fmodelfile hklout cadded.mtz << EOF > /dev/null
labin file 1 E1=FMODEL E2=PHIFMODEL
labin file 2 E1=FOBS E2=SIGFOBS E3=R_FREE_FLAGS
EOF
if($status) then
    set BAD = "garbled column labels in $fmodelfile   note that anomalous refinement is not supported! "
    goto exit
endif
scaleit hklin cadded.mtz hklout kickme.mtz << EOF >! scaleit_fobs.log
refine anisotropic
labin FP=FMODEL SIGFP=SIGFOBS FPH1=FOBS SIGFPH1=SIGFOBS
EOF
set fobs_scale = `awk '/Derivative   1  /{print $3+0}' scaleit_fobs.log | tail -1`
set fobs_B     = `awk '/The equivalent isotropic temperature factor is/{print $7}' scaleit_fobs.log | tail -1`
echo "scale factor: $fobs_scale $fobs_B"



echo "overall F000 = $F000 +/- $SIGF000"
echo "2FoFc_END.map is on an absolute electron number-density scale (electrons/A^3)"
echo "FoFc_scaled.map is on the same scale"

























if($?NO_RAPID) goto exit


# check if we need map_rmsd.com
which map_rmsd.com >& /dev/null
if($status) then
    goto deploy_map_rmsd
endif
got_map_rmsd:


if($?NO_FOFC_RAPID) goto SIGF_RAPID

echo ""
echo ""
echo "computing RAPID map using Fo-Fc as the error to propagate"

set test = `cat seeds.txt |& wc -l`
if(! -e seeds.txt || $test < $RAPID_itrs) then
    echo "generating default random-number seeds in seeds.txt"
    echo $RAPID_itrs | awk '{for(i=1;i<=3*$1;++i) print i}' >! seeds.txt
endif

if(! -s ${prefix}_001.pdb && -e ${prefix}_1.pdb) cp -p ${prefix}_1.pdb ${prefix}_001.pdb
grep -v hexdigest ${prefix}_001.pdb >! refme.pdb

# check if we need kick_data_bydiff.com
which kick_data_bydiff.com >& /dev/null
if($status) then
    goto deploy_kick_data_bydiff
endif
got_kick_data_bydiff:


set cpu = 0
foreach seed ( `head -n $RAPID_itrs seeds.txt` )
   @ cpu = ( $cpu + 1 )

    kick_data_bydiff.com kickme.mtz FC=FMODEL seed=$seed
    if($status) then
	set BAD = "kick_data_bydiff.com failed."
	goto exit
    endif
    cad hklin1 kicked.mtz hklout refme_FOFC_${seed}.mtz << EOF > /dev/null
    labin file 1 E1=FOBS E2=SIGFOBS E3=R_FREE_FLAGS
EOF

    phenix.refine ${prefix}.eff refme.pdb refme_FOFC_${seed}.mtz \
       export_final_f_model=True \
       write_geo_file=False write_def_file=False \
       main.number_of_macro_cycles=$cycles \
       output.prefix=seed_${seed} >&! seed_${seed}.log &

    if($cpu >= $CPUs) then
        wait
        set cpu = 0
    endif
end

wait

foreach seed ( `head -n $RAPID_itrs seeds.txt` )

    echo "seed $seed"

    tail -n 2 seed_${seed}.log

    set mtzout = seed_${seed}_001.mtz
    if(! -e "$mtzout") set mtzout = seed_${seed}_001_map_coeffs.mtz
    if(! -e "$mtzout") set mtzout = seed_${seed}_1_map_coeffs.mtz
    if(! -e "$mtzout") set mtzout = seed_${seed}_001.mtz
    if(! -e "$mtzout") set mtzout = seed_${seed}_1.mtz

    # make sure this is on an absolute scale
#rm -f fmodel_${seed}.mtz
#phenix.fmodel seed_${seed}_001.pdb k_sol=$ksol b_sol=$Bsol high_res=$phenix_reso \
#  file_name=fmodel_${seed}.mtz $extra >! phenix_fmodel.log

    cad hklin1 fmodel.mtz hklin2 $mtzout hklout cadded.mtz << EOF > /dev/null
    labin file 1 E1=FMODEL E2=PHIFMODEL
    labin file 2 E1=2FOFCWT E2=PH2FOFCWT
EOF
    scaleit hklin cadded.mtz hklout scaled.mtz << EOF >! scaleit.log
    labin FP=FMODEL SIGFP=FMODEL FPH1=2FOFCWT SIGFPH1=FMODEL
    refine isotropic
EOF
    set scale = `awk '/Derivative   1  /{print $3+0}' scaleit.log | tail -1`
    echo "scale factor: $scale"

    fft hklin $mtzout mapout ${tempfile}ffted.map << EOF > /dev/null
    labin F1=2FOFCWT PHI=PH2FOFCWT
    scale F1 $scale
    grid $xyzgrid
EOF
    echo "xyzlim asu" |\
     mapmask mapin ${tempfile}ffted.map mapout seed_${seed}_2FoFc.map > /dev/null

    fft hklin $mtzout mapout ${tempfile}ffted.map << EOF > /dev/null
    labin F1=FOFCWT PHI=PHFOFCWT
    scale F1 $scale
    grid $xyzgrid
EOF
    echo "xyzlim asu" |\
     mapmask mapin ${tempfile}ffted.map mapout seed_${seed}_FoFc.map > /dev/null
    rm -f ${tempfile}ffted.map

end





map_rmsd.com seed*_2FoFc.map ref=2FoFc_scaled.map | grep dens
mv sigma.map 2FoFc_error.map
if(! -s 2FoFc_error.map) then
    set BAD = "map_rmsd.com failed.  Is it installed? "
    goto exit
endif
echo "2FoFc_error.map is the RAPID map of error bars propagated from Fobs-Fcalc"

map_rmsd.com seed*_FoFc.map ref=FoFc_scaled.map | grep dens
mv sigma.map FoFc_error.map
if(! -s FoFc_error.map) then
    set BAD = "map_rmsd.com failed.  Is it installed? "
    goto exit
endif
echo "FoFc_error.map is the RAPID map of error bars propagated from Fobs-Fcalc"





# now make a signal-to-noise map?
# mapsig can divide maps, but only whole-cell maps...
echo xyzlim cell |\
  mapmask mapin1 2FoFc_END.map mapout ${tempfile}rho.map > /dev/null
echo xyzlim cell |\
  mapmask mapin1 2FoFc_error.map mapout ${tempfile}sigrho.map > /dev/null

mapsig mapin ${tempfile}rho.map mapin2 ${tempfile}sigrho.map MAPOUT ${tempfile}snr.map << EOF > /dev/null
TYPE RATIO
MAPOUT
EOF
# mapsig multiplies by 100 for some reason
mapmask mapin ${tempfile}snr.map mapout 2FoFc_snr.map << EOF > /dev/null
SCALE FACTOR 0.01
xyzlim ASU
EOF
rm -f ${tempfile}rho.map > /dev/null
rm -f ${tempfile}sigrho.map > /dev/null
rm -f ${tempfile}snr.map > /dev/null




# now make a signal-to-noise map?
# mapsig can divide maps, but only whole-cell maps...
echo xyzlim cell |\
  mapmask mapin1 FoFc_scaled.map mapout ${tempfile}rho.map > /dev/null
echo xyzlim cell |\
  mapmask mapin1 FoFc_error.map mapout ${tempfile}sigrho.map > /dev/null

mapsig mapin ${tempfile}rho.map mapin2 ${tempfile}sigrho.map MAPOUT ${tempfile}snr.map << EOF > /dev/null
TYPE RATIO
MAPOUT
EOF
# mapsig multiplies by 100 for some reason
mapmask mapin ${tempfile}snr.map mapout FoFc_snr.map << EOF > /dev/null
SCALE FACTOR 0.01
xyzlim ASU
EOF
rm -f ${tempfile}rho.map > /dev/null
rm -f ${tempfile}sigrho.map > /dev/null
rm -f ${tempfile}snr.map > /dev/null








SIGF_RAPID:

if($?NO_SIGF_RAPID) goto exit


echo ""
echo ""
echo "computing RAPID map using sigF as the error to propagate"


set test = `cat seeds.txt |& wc -l`
if(! -e seeds.txt || $test < $RAPID_itrs) then
    echo "generating default random-number seeds in seeds.txt"
    echo $RAPID_itrs | awk '{for(i=1;i<=3*$1;++i) print i}' >! seeds.txt
endif

if(! -s ${prefix}_001.pdb && -e ${prefix}_1.pdb) cp -p ${prefix}_1.pdb ${prefix}_001.pdb
grep -v hexdigest ${prefix}_001.pdb >! refme.pdb


# check if we need kick_data_bydiff.com
which kick_data.com >& /dev/null
if($status) then
    goto deploy_kick_data
endif
got_kick_data:



set cpu = 0
foreach seed ( `head -n $RAPID_itrs seeds.txt` )
   @ cpu = ( $cpu + 1 )

    kick_data.com kickme.mtz seed=$seed
    if($status) then
	set BAD = "kick_data.com failed."
	goto exit
    endif
    cad hklin1 kicked.mtz hklout refme_sigF_${seed}.mtz << EOF > /dev/null
    labin file 1 E1=FOBS E2=SIGFOBS E3=R_FREE_FLAGS
EOF

    phenix.refine ${prefix}.eff refme.pdb refme_sigF_${seed}.mtz \
       export_final_f_model=True \
       write_geo_file=False write_def_file=False \
       main.number_of_macro_cycles=$cycles \
       output.prefix=sigF_seed_${seed} >&! sigF_seed_${seed}.log &

    if($cpu >= $CPUs) then
        wait
        set cpu = 0
    endif
end

wait

foreach seed ( `head -n $RAPID_itrs seeds.txt` )

    tail -2 sigF_seed_${seed}.log

    set mtzout = sigF_seed_${seed}_001_map_coeffs.mtz
    if(! -e "$mtzout") set mtzout = sigF_seed_${seed}_1_map_coeffs.mtz
    if(! -e "$mtzout") set mtzout = sigF_seed_${seed}_1.mtz
    if(! -e "$mtzout") set mtzout = sigF_seed_${seed}_001.mtz

    # make sure this is on an absolute scale
    cad hklin1 fmodel.mtz hklin2 $mtzout hklout cadded.mtz << EOF > /dev/null
    labin file 1 E1=FMODEL E2=PHIFMODEL
    labin file 2 E1=2FOFCWT E2=PH2FOFCWT
EOF
    scaleit hklin cadded.mtz hklout scaled.mtz << EOF >! scaleit.log
    labin FP=FMODEL SIGFP=FMODEL FPH1=2FOFCWT SIGFPH1=FMODEL
EOF
    set scale = `awk '/Derivative   1  /{print $3+0}' scaleit.log | tail -1`
    echo "scale factor: $scale"

    fft hklin $mtzout mapout ${tempfile}ffted.map << EOF > /dev/null
    labin F1=2FOFCWT PHI=PH2FOFCWT
    scale F1 $scale
    grid $xyzgrid
EOF
    echo "xyzlim asu" |\
     mapmask mapin ${tempfile}ffted.map mapout sigF_seed_${seed}_2FoFc.map > /dev/null

    fft hklin $mtzout mapout ${tempfile}ffted.map << EOF > /dev/null
    labin F1=FOFCWT PHI=PHFOFCWT
    scale F1 $scale
    grid $xyzgrid
EOF
    echo "xyzlim asu" |\
     mapmask mapin ${tempfile}ffted.map mapout sigF_seed_${seed}_FoFc.map > /dev/null
    rm -f ${tempfile}ffted.map

end


map_rmsd.com sigF_seed*_2FoFc.map ref=2FoFc_scaled.map | grep dens
mv sigma.map 2FoFc_sigF_error.map
if(! -s 2FoFc_sigF_error.map) then
    set BAD = "map_rmsd.com failed.  Is it installed? "
    goto exit
endif
echo "2FoFc_sigF_error.map is the RAPID map of error bars propagated from sigF"

map_rmsd.com sigF_seed*_FoFc.map ref=FoFc_scaled.map | grep dens
mv sigma.map FoFc_sigF_error.map
if(! -s FoFc_sigF_error.map) then
    set BAD = "map_rmsd.com failed.  Is it installed? "
    goto exit
endif
echo "FoFc_sigF_error.map is the RAPID map of error bars propagated from sigF"





# now make a signal-to-noise map?
# mapsig can divide maps, but only whole-cell maps...
echo xyzlim cell |\
  mapmask mapin1 2FoFc_END.map mapout ${tempfile}rho.map > /dev/null
echo xyzlim cell |\
  mapmask mapin1 2FoFc_sigF_error.map mapout ${tempfile}sigrho.map > /dev/null

mapsig mapin ${tempfile}rho.map mapin2 ${tempfile}sigrho.map MAPOUT ${tempfile}snr.map << EOF > /dev/null
TYPE RATIO
MAPOUT
EOF
# mapsig multiplies by 100 for some reason
mapmask mapin ${tempfile}snr.map mapout 2FoFc_sigF_snr.map << EOF > /dev/null
SCALE FACTOR 0.01
xyzlim ASU
EOF
rm -f ${tempfile}rho.map > /dev/null
rm -f ${tempfile}sigrho.map > /dev/null
rm -f ${tempfile}snr.map > /dev/null




# now make a signal-to-noise map?
# mapsig can divide maps, but only whole-cell maps...
echo xyzlim cell |\
  mapmask mapin1 FoFc_scaled.map mapout ${tempfile}rho.map > /dev/null
echo xyzlim cell |\
  mapmask mapin1 FoFc_sigF_error.map mapout ${tempfile}sigrho.map > /dev/null

mapsig mapin ${tempfile}rho.map mapin2 ${tempfile}sigrho.map MAPOUT ${tempfile}snr.map << EOF > /dev/null
TYPE RATIO
MAPOUT
EOF
# mapsig multiplies by 100 for some reason
mapmask mapin ${tempfile}snr.map mapout FoFc_sigF_snr.map << EOF > /dev/null
SCALE FACTOR 0.01
xyzlim ASU
EOF
rm -f ${tempfile}rho.map > /dev/null
rm -f ${tempfile}sigrho.map > /dev/null
rm -f ${tempfile}snr.map > /dev/null


if(-e 2FoFc_END.map) echo "2FoFc_END.map is on an absolute electron number-density scale"
if(-e 2FoFc_error.map) echo "2FoFc_error.map is the RAPID map of error bars propagated from Fobs-Fcalc"
if(-e FoFc_error.map) echo "FoFc_error.map is the RAPID map of error bars propagated from Fobs-Fcalc"
if(-e 2FoFc_sigF_error.map) echo "2FoFc_sigF_error.map is the RAPID map of error bars propagated from sigFobs"
if(-e FoFc_sigF_error.map) echo "FoFc_sigF_error.map is the RAPID map of error bars propagated from sigFobs"
if(-e 2FoFc_snr.map) echo "2FoFc_snr.map is the ratio of 2FoFc_END.map to 2FoFc_error.map"
if(-e FoFc_snr.map) echo "FoFc_snr.map is the ratio of FoFc_scaled.map to FoFc_error.map"
if(-e 2FoFc_sigF_snr.map) echo "2FoFc_sigF_snr.map is the ratio of 2FoFc_END.map to 2FoFc_sigF_error.map"
if(-e FoFc_sigF_snr.map) echo "FoFc_sigF_snr.map is the ratio of FoFc_scaled.map to 2FoFc_sigF_error.map"





exit:
if($?BAD) then
    echo "ERROR: $BAD"
    cat << EOF
usage:
$0 phenixrefine.eff 

where:
phenixrefine.eff  - is the name of the "eff" file produced by phenix.refine at your
                    last round of refinement
other options:
ksol=x            - manually set the bulk solvent density to "x" electrons/A^3
Rsol=x            - manually set the solvent radius for phenix.refine
Rshrink=x         - manually set the solvent "shrink" radius for phenix.refine

cycles=n          - number of macro cycles for each phenix.refine job
seeds=n           - do "n" parallel refinements for the RAPID maps (default: $RAPID_itrs)
cpus=n            - use up to "n" CPUs to do parallel refinements for RAPID maps

-nofofc           - do not calculate the RAPID map of Fo-Fc errors
-nosigf           - do not calculate the RAPID map of sigma(F) errors
-norapid          - do not calculate any RAPID noise maps
EOF
    exit 9
endif

exit



deploy_map_vacuum_level:

echo "deploying script: map_vacuum_level.com"
cat << EOF-script >! map_vacuum_level.com
#! /bin/tcsh -f
#
#	try to find the "vacuum level" of the provided electron
#	density map using "robust statistics"
#
#
set map = "\$1"

set tempfile = \${CCP4_SCR}/vacuum\$\$


# make one ASU
echo "xyzlim ASU" | mapmask mapin \$map mapout \${tempfile}test.map > /dev/null
if(\$status) then
    echo "something is wrong with map: \$map"
    exit 9
endif


# examine this map to get the grid and axis convention for this SG
echo "" | mapdump mapin \${tempfile}test.map >! \${tempfile}maphead.txt
set xyzgrid = \`awk '/Grid sampling on x, y, z ../{print \$8,\$9,\$10}' \${tempfile}maphead.txt  | tail -1\`
set voxels = \`awk '/Number of columns, rows, sections/{print \$7*\$8*\$9}' \${tempfile}maphead.txt  | tail -1\`
#set mean   = \`awk '/Mean density/{print \$NF}' \${tempfile}maphead.txt\`

set CELL = \`awk '/Cell dimensions ../{print \$4,\$5,\$6,\$7,\$8,\$9}' \${tempfile}maphead.txt\`
echo \$CELL |\\
awk 'NF==6{DTR=atan2(1,1)/45; A=cos(DTR*\$4); B=cos(DTR*\$5); G=cos(DTR*\$6); \\
 skew = 1 + 2*A*B*G - A*A - B*B - G*G ; if(skew < 0) skew = -skew;\\
 printf "%.3f\\n", \$1*\$2*\$3*sqrt(skew)}' |\\
cat >! \${tempfile}volume
set CELLvolume = \`cat \${tempfile}volume\`
rm -f \${tempfile}volume
rm -f \${tempfile}maphead.txt


# convert voxel values to text
set datasize = \`echo \$voxels | awk '{print 4*\$1}'\`
set filesize = \`ls -l \${tempfile}test.map | awk '{print \$5}'\`
set head = \`echo "\$filesize \$datasize" | awk '{print \$1-\$2}'\`
echo "reading \$map"
od -vf -w4 -j \$head \${tempfile}test.map | awk 'NF==2{print \$2}' > ! \${tempfile}map.txt
set mean = \`awk '{++n;sum+=\$1} END{print sum/n}' \${tempfile}map.txt\`

echo "cell volume is : \$CELLvolume    mean electron density: \$mean electrons/A^3"

if (\$?histograms) then
    ~jamesh/awk/histogram.awk -v bs=0.01 \${tempfile}map.txt | sort -g >! hist0.txt
endif


# start by looking only at negative side
set cut = \`echo \$mean | awk '\$1<=0{\$1=1e-99} {print}'\`
set mad = 1e-6

awk -v cut=\$cut '\$1<cut' \${tempfile}map.txt |\\
sort -g >! \${tempfile}sorted.txt

set median = \$cut
set lastmedian = 1e99
echo "symmetric median rejection"
while ( \$median != \$lastmedian ) 
    cat \${tempfile}sorted.txt |\\
    awk -v cut=\$cut '{print} \$1>cut{exit}' |\\
    tee \${tempfile}new.txt |\\
    awk 'NR==1{min=\$1} {++n;v[n]=\$1}\\
     END{print min+0,v[int(n/2)]+0,n}' >! \${tempfile}median.txt

    set cut = \`awk '{print \$1+2*(\$2-\$1)}' \${tempfile}median.txt\`
    set n = \`awk '{print \$3}' \${tempfile}median.txt\`
    set frac = \`echo \$n \$voxels | awk '{printf "%d",\$1/\$2*100}'\`
    if(\$frac < 1) goto finish
    set lastmedian = \$median
    set median = \`awk '{print \$2}' \${tempfile}median.txt\`
    set min = \`head -n 1 \${tempfile}sorted.txt | awk '{print \$1+0}'\`
    echo "\$map median= \$median   cut= \$cut   frac= \${frac}%   min= \$min"
    set median = \$min
    mv \${tempfile}new.txt \${tempfile}sorted.txt
end
# now use 3x median absolute deviation as upper rejection cutoff
foreach sig ( 4 3  )
    echo "+\$sig x MAD rejection"
set lastmedian = 1e99
while ( \$median != \$lastmedian ) 
    cat \${tempfile}sorted.txt |\\
    awk -v median=\$median '{print sqrt((\$1-median)^2)}' |\\
    sort -g | awk '{++n;v[n]=\$1}\\
     END{print v[int(n/2)]}' >! \${tempfile}mad.txt
    set mad = \`cat \${tempfile}mad.txt\`

    cat \${tempfile}sorted.txt |\\
    awk -v median=\$median -v mad=\$mad -v sig=\$sig '\\
	\$1 <= median+sig*mad' |\\
    tee \${tempfile}new.txt |\\
    awk '{++n;v[n]=\$1}\\
     END{print v[int(n/2)]+0,n}' >! \${tempfile}median.txt

    set n = \`awk '{print \$2}' \${tempfile}median.txt\`
    set frac = \`echo \$n \$voxels | awk '{printf "%d",\$1/\$2*100}'\`
    if(\$frac < 1) goto finish
    set min = \`head -n 1 \${tempfile}sorted.txt | awk '{print \$1+0}'\`
    set lastmedian = \$median
    set median = \`awk '{print \$1}' \${tempfile}median.txt\`
    echo "\$map median= \$median   mad= \$mad   frac= \${frac}%   min= \$min"
    mv \${tempfile}new.txt \${tempfile}sorted.txt
end
end
# now use 2x median absolute deviation as upper and lower rejection cutoff
foreach sig ( 4 3  )
    echo "+/- \$sig x MAD rejection"
set lastmedian = 1e99
while ( \$median != \$lastmedian ) 
    cat \${tempfile}sorted.txt |\\
    awk -v median=\$median '{print sqrt((\$1-median)^2)}' |\\
    sort -g | awk '{++n;v[n]=\$1}\\
     END{print v[int(n/2)]+0}' >! \${tempfile}mad.txt
    set mad = \`cat \${tempfile}mad.txt\`

    cat \${tempfile}sorted.txt |\\
    awk -v median=\$median -v mad=\$mad -v sig=\$sig '\\
	sqrt((\$1-median)^2) <= sig*mad' |\\
    tee \${tempfile}new.txt |\\
    awk '{++n;v[n]=\$1}\\
     END{print v[int(n/2)]+0,n}' >! \${tempfile}median.txt

    set n = \`awk '{print \$2}' \${tempfile}median.txt\`
    set frac = \`echo \$n \$voxels | awk '{printf "%d",\$1/\$2*100}'\`
    if(\$frac < 1) goto finish
    set min = \`head -n 1 \${tempfile}sorted.txt | awk '{print \$1+0}'\`
    set lastmedian = \$median
    set median = \`awk '{print \$1}' \${tempfile}median.txt\`
    echo "\$map median= \$median   mad= \$mad   frac= \${frac}%   min= \$min"
    mv \${tempfile}new.txt \${tempfile}sorted.txt
end
end
finish:
set vacuum = \`awk '{++n;sum+=\$1} END{print sum/n}' \${tempfile}sorted.txt\`
echo "\$map vacuum level: \$vacuum +/- \$mad  occupies \${frac}% of map"

if(\$?histograms) then
    set bs = \`echo \$mad | awk '\$1==0{\$1=1e-6} {print \$1/10}'\`
    awk -v vacuum=\$vacuum '{print \$1-vacuum}' \${tempfile}sorted.txt |\\
    ~jamesh/awk/histogram.awk -v bs=\$bs |\\
      sort -g >! baseline_hist.txt
    awk -v vacuum=\$vacuum '{print \$1-vacuum}' \${tempfile}map.txt |\\
    ~jamesh/awk/histogram.awk -v bs=\$bs |\\
      sort -g >! hist.txt
    cat \${tempfile}map.txt |\\
    ~jamesh/awk/histogram.awk -v bs=\$bs |\\
      sort -g >! hist0.txt
endif

set mean = \`awk '{++n;sum+=\$1} END{print sum/n}' \${tempfile}map.txt\`
set F000 = \`echo \$CELLvolume \$mean \$vacuum | awk '{print \$1*(\$2-\$3)}'\`
echo "estimated F000 = \$F000"

set offset = \`echo \$vacuum | awk '{print -\$1}'\`
mapmask mapin \${tempfile}test.map mapout vacuum_zero.map << EOF > /dev/null
scale factr 1 \$offset
EOF
echo "vacuum_zero.map has a vacuum level of zero"


rm -f \${tempfile}test.map
rm -f \${tempfile}map.txt
rm -f \${tempfile}sorted.txt
rm -f \${tempfile}median.txt
rm -f \${tempfile}mad.txt


exit

EOF-script
chmod a+x map_vacuum_level.com


set path = ( . $path )
rehash
goto got_map_vacuum_level










deploy_kick_data_bydiff:

echo "deploying script: kick_data_bydiff.com"
cat << EOF-script >! kick_data_bydiff.com
#! /bin/tcsh -f
#
#	change FP by RMS (FP - FC)				-James Holton 6-24-12
#
#


set mtzfile = refined.mtz
set F = ""
set SIGF = ""
set FC = ""
set seed = \`date +%N | awk '{print \$1/1000}'\`


set tempfile = \${CCP4_SCR}/kick_data\$\$


foreach arg ( \$* )

    if( "\$arg" =~ *.mtz ) set mtzfile  = "\$arg"
    if(( "\$arg" =~ *[0-9] )&&( "\$arg" =~ [1-9]* )) set steps = "\$arg"

    if( "\$arg" =~ seed=* ) then
        set test = \`echo \$arg | awk -F "[=]" '\$2+0>0{print \$2+0}'\`
        if("\$test" != "") set seed = \$test
    endif
    if( "\$arg" =~ F=* ) then
        set test = \`echo \$arg | awk -F "[=]" '{print \$2}'\`
        if("\$test" != "") set user_F = \$test
    endif
    if( "\$arg" =~ FC=* ) then
        set test = \`echo \$arg | awk -F "[=]" '{print \$2}'\`
        if("\$test" != "") set user_FC = \$test
    endif
end

# examine MTZ file
echo "go" | mtzdump hklin \$mtzfile |\\
awk '/OVERALL FILE STATISTICS/,/No. of reflections used/' |\\
awk 'NF>5 && \$(NF-1) ~ /^[FJDGKQLMPWABYIRUV]\$/' |\\
cat >! \${tempfile}mtzdmp

# use completeness, or F/sigF to pick default F
cat \${tempfile}mtzdmp |\\
awk '\$(NF-1) == "F"{F=\$NF; meanF=\$8; reso=\$(NF-2); comp=substr(\$0,32)+0; \\
  getline; if(\$(NF-1) != "Q") next; \\
  S=\$NF; if(\$8) meanF /= \$8; print F, S, reso, comp, meanF;}' |\\
sort -k3n,4 -k4nr,5 -k5nr >! \${tempfile}F

cat \${tempfile}mtzdmp |\\
awk '\$(NF-1) == "F"{F=\$NF; meanF=\$8; reso=\$(NF-2); comp=substr(\$0,32)+0; \\
  getline; if(\$(NF-1) != "P") next; \\
  PHI=\$NF; if(\$8) meanF /= \$8; print F, PHI, reso, comp, meanF;}' |\\
cat >! \${tempfile}FC

# and extract all dataset types/labels
cat \${tempfile}mtzdmp |\\
awk 'NF>2{print \$(NF-1), \$NF, " "}' |\\
cat >! \${tempfile}cards

#clean up
rm -f \${tempfile}mtzdmp

if("\$F" == "" || "\$SIGF" == "") then
    # pick F with best resolution, or F/sigma
    if(\$?user_F) then
	set F = \`awk -v F=\$user_F '\$1==F{print}' \${tempfile}F\`
	if("\$F" == "") then
	    echo "WARNING: \$user_F not found in \$mtzfile"
	    cat \${tempfile}cards
	endif
    endif
    if(\$#F < 2) set F = \`head -1 \${tempfile}F\`
    if(\$#F > 2) then
	set SIGF = \$F[2]
	set F    = \$F[1]
	echo "selected F=\$F SIGF=\$SIGF "
    endif
    rm -f \${tempfile}F
endif
if("\$F" == "") then
    set BAD = "no Fobs in \$mtzfile "
    goto exit
endif
if("\$FC" == "") then
    # pick FC with best resolution?  Or just first one in the file.
    if(\$?user_FC) then
	set FC = \`awk -v FC=\$user_FC '\$1==FC{print}' \${tempfile}FC\`
	if("\$FC" == "") set FC = \`awk -v FC=\$user_FC '\$2==FC{print \$2}' \${tempfile}cards\`
	if("\$FC" == "") then
	    echo "WARNING: \$user_FC not found in \$mtzfile"
	    cat \${tempfile}cards
	endif
    endif
    if("\$FC" == "") set FC = \`head -1 \${tempfile}FC\`
    if(\$#FC > 1) then
	set FC    = \$FC[1]
	echo "selected FC=\$FC "
    endif
    rm -f \${tempfile}FC
endif
if("\$FC" == "") then
    set BAD = "no Fcalc in \$mtzfile "
    goto exit
else
    echo "using FC=\$FC"
endif
# capture remaining columns for imort later...
set otherstuff = \`awk -v F=\$F -v SIGF=\$SIGF -v FC=\$FC '\$2!=F && \$2!=SIGF && \$2!=FC{++n;print "E"n"="\$2}' \${tempfile}cards\`
rm -f \${tempfile}cards
#echo "\$otherstuff"


# extract only F/SIGF so sftools does not get pissed off
cad hklin1 \$mtzfile hklout F_FC.mtz << EOF > /dev/null
labin file 1 E1=\$F E2=\$SIGF E3=\$FC
EOF

echo "using seed = \$seed"
sftools << EOF >&! \${tempfile}sftools.log
read F_FC.mtz
calc seed \$seed
calc col delta = col \$F col \$FC -
calc col noise = col delta RAN_G *
calc F col Fnew  = col \$F col noise +
select col Fnew < 0
calc col Fnew = 0
select all
calc F col \$F = col Fnew
delete  col Fnew noise delta
write new.mtz 
y
exit
y
EOF
if(\$status) then
    set BAD = "sftools failed"
    goto exit
endif

# one last clean-up
mv new.mtz cleanme.mtz
sftools << EOF > /dev/null
read cleanme.mtz
absent col \$SIGF if col \$F ABSENT
write new.mtz
EOF
cad hklin1 new.mtz hklout cleaned.mtz << EOF > /dev/null
labin file 1 all
EOF


# now put other stuff back in
if("\$otherstuff" != "") then
    cad hklin1 cleaned.mtz hklin2 \$mtzfile hklout kicked.mtz << EOF > /dev/null
labin file 1 all
labin file 2 \$otherstuff
EOF
else
    cad hklin1 cleaned.mtz hklout kicked.mtz << EOF > /dev/null
labin file 1 all
EOF
endif
if(\$status) then
    set BAD = "output mtz corrupted."
    goto exit
endif
rm -f F_FC.mtz
rm -f new.mtz
rm -f cleaned.mtz

echo "kicked.mtz contains \$F from \$mtzfile modified by rms ( \$F - \$FC )"

exit:
if(\$?BAD) then
    echo "ERROR: \$BAD"
    exit 9
endif

exit

EOF-script
chmod a+x kick_data_bydiff.com


set path = ( . $path )
rehash
goto got_kick_data_bydiff
















deploy_kick_data:

echo "deploying script: kick_data.com"
cat << EOF-script >! kick_data.com
#! /bin/tcsh -f
#
#	change FP by RMS SIGFP				-James Holton  3-2-12
#
#


set mtzfile = refme.mtz
set F = ""
set SIGF = ""
set seed = \`date +%N | awk '{print \$1/1000}'\`


set tempfile = \${CCP4_SCR}/kick_data\$\$


foreach arg ( \$* )

    if( "\$arg" =~ *.mtz ) set mtzfile  = "\$arg"
    if(( "\$arg" =~ *[0-9] )&&( "\$arg" =~ [1-9]* )) set steps = "\$arg"

    if( "\$arg" =~ seed=* ) then
        set test = \`echo \$arg | awk -F "[=]" '\$2+0>0{print \$2+0}'\`
        if("\$test" != "") set seed = \$test
    endif
    if( "\$arg" =~ F=* ) then
        set test = \`echo \$arg | awk -F "[=]" '\$2+0>0{print \$2+0}'\`
        if("\$test" != "") set user_F = \$test
    endif
end

if(! -e "\$mtzfile") then
    set BAD = "\$mtzfile does not exist"
    goto exit
endif

# examine MTZ file
echo "go" | mtzdump hklin \$mtzfile |\\
awk '/OVERALL FILE STATISTICS/,/No. of reflections used/' |\\
awk 'NF>5 && \$(NF-1) ~ /^[FJDGKQLMPWABYIRUV]\$/' |\\
cat >! \${tempfile}mtzdmp

# use completeness, or F/sigF to pick default F
cat \${tempfile}mtzdmp |\\
awk '\$(NF-1) == "F"{F=\$NF; meanF=\$8; reso=\$(NF-2); comp=substr(\$0,32)+0; \\
  getline; if(\$(NF-1) != "Q") next; \\
  S=\$NF; if(\$8) meanF /= \$8; print F, S, reso, comp, meanF;}' |\\
sort -k3n,4 -k4nr,5 -k5nr >! \${tempfile}F

# and extract all dataset types/labels
cat \${tempfile}mtzdmp |\\
awk 'NF>2{print \$(NF-1), \$NF, " "}' |\\
cat >! \${tempfile}cards

#clean up
rm -f \${tempfile}mtzdmp

if("\$F" == "" || "\$SIGF" == "") then
    # pick F with best resolution, or F/sigma
    if(\$?user_F) then
	set F = \`awk -v F=\$user_F '\$1==F{print}' \${tempfile}F\`
    endif
    if(\$#F < 2) set F = \`head -1 \${tempfile}F\`
    if(\$#F > 2) then
	set SIGF = \$F[2]
	set F    = \$F[1]
	echo "selected F=\$F SIGF=\$SIGF "
    endif
    rm -f \${tempfile}F
endif
# capture remaining columns for imort later...
set otherstuff = \`awk -v F=\$F -v SIGF=\$SIGF '\$2!=F && \$2!=SIGF{++n;print "E"n"="\$2}' \${tempfile}cards\`
rm -f \${tempfile}cards
#echo "\$otherstuff"

# extract only F/SIGF so sftools does not get pissed off
cad hklin1 \$mtzfile hklout F_SIGF.mtz << EOF > /dev/null
labin file 1 E1=\$F E2=\$SIGF
EOF

echo "using seed = \$seed"
sftools << EOF >&! \${tempfile}sftools.log
read F_SIGF.mtz
calc seed \$seed
calc col noise = col \$SIGF RAN_G *
calc F col Fnew  = col \$F col noise +
select col Fnew < 0
calc col Fnew = 0
select all
calc F col \$F = col Fnew
delete  col Fnew noise
write new.mtz 
y
exit
y
EOF
if(\$status) then
    set BAD = "sftools failed"
    goto exit
endif
if("\$otherstuff" != "") then
    cad hklin1 new.mtz hklin2 \$mtzfile hklout kicked.mtz << EOF > /dev/null
labin file 1 all
labin file 2 \$otherstuff
EOF
else
    cad hklin1 new.mtz hklout kicked.mtz << EOF > /dev/null
labin file 1 all
EOF
endif
if(\$status) then
    set BAD = "output mtz corrupted."
    goto exit
endif
rm -f F_SIGF.mtz
rm -f new.mtz

echo "kicked.mtz contains \$F from \$mtzfile modified by rms \$SIGF"

exit:
if(\$?BAD) then
    echo "ERROR: \$BAD"
    exit 9
endif

exit

EOF-script
chmod a+x kick_data.com


set path = ( . $path )
rehash
goto got_kick_data







deploy_map_rmsd:

echo "deploying script: map_rmsd.com"
cat << EOF-script >! map_rmsd.com
#! /bin/tcsh -f
#
#
#	compute the average and variance of a series of maps
#
#	uses auxillary program float_mult				-James Holton 1-20-12
#
#
set tempfile = \${CCP4_SCR}/mapvar\$\$

set ref = ""
set maps = ""
foreach arg ( \$* )
    
    if("\$arg" =~ ref=*) then
	set ref = \`echo \$arg | awk '{print substr(\$0,5)}'\`
    else
	set maps = ( \$maps \$arg )
    endif
end

if(\$#maps == 0) then
    echo "usage: \$0 *.map"
    exit 9
endif
if(-e "\$ref") goto got_ref

echo "scale factor 0 0" |\\
  mapmask mapin \$maps[1] mapout \${tempfile}sum.map > /dev/null

foreach map ( \$maps )
    echo "\$map"

    echo "maps add" |\\
    mapmask mapin1 \$map mapin2 \${tempfile}sum.map mapout \${tempfile}temp.map > /dev/null
    mv \${tempfile}temp.map \${tempfile}sum.map
end
set scale = \`echo \$#maps | awk '{print 1/\$1}'\`
echo "scale factor \$scale 0" |\\
  mapmask mapin \${tempfile}sum.map mapout avg.map > /dev/null
set ref = avg.map
echo "avg.map is the average of all maps"

got_ref:
echo "scale factor -1 0" |\\
  mapmask mapin \$ref mapout \${tempfile}negavg.map > /dev/null
echo "scale factor 0 0" |\\
  mapmask mapin \$ref mapout \${tempfile}sum.map > /dev/null

foreach map ( \$maps )
    echo "\$map - \$ref"

    echo "maps add" |\\
    mapmask mapin1 \$map mapin2 \${tempfile}negavg.map mapout \${tempfile}temp.map > /dev/null
    echo "maps mult" |\\
    mapmask mapin1 \${tempfile}temp.map mapin2 \${tempfile}temp.map mapout \${tempfile}sqrdiff.map > /dev/null
    echo "maps add" |\\
    mapmask mapin1 \${tempfile}sum.map mapin2 \${tempfile}sqrdiff.map mapout \${tempfile}temp.map > /dev/null
    mv \${tempfile}temp.map \${tempfile}sum.map
end
if(\$#maps != 1) then
    set scale = \`echo \$#maps | awk '{print 1/(\$1-1)}'\`
else
    echo "WARNING: only one map! output will simply be absolute difference."
    set scale = 1
endif
echo "scale factor \$scale 0" |\\
  mapmask mapin \${tempfile}sum.map mapout variance.map > /dev/null


rm -f \${tempfile}temp.map \${tempfile}sum.map \${tempfile}sqrdiff.map \${tempfile}negavg.map

echo "variance.map is the variance from \$ref"

echo go | mapdump mapin variance.map | tee \${tempfile}mapdump.log |\\
awk '/Grid sampling on x, y, z/{gx=\$8;gy=\$9;gz=\$10}\\
     /Maximum density /{max=\$NF}\\
     /Cell dimensions /{xs=\$4/gx;ys=\$5/gy;zs=\$6/gz}\\
     /Number of columns, rows, sections/{nc=\$7;nr=\$8;ns=\$9}\\
 END{print xs,ys,zs,nc,nr,ns,max}' >! \${tempfile}mapstuff.txt

set size = \`awk '{print 4*\$4*\$5*\$6}' \${tempfile}mapstuff.txt\`
set head = \`ls -l variance.map | awk -v size=\$size '{print \$5-size}'\`
set skip = \`echo \$head | awk '{print \$1+1}'\`


# now somehow take the square root?
sqrt:
if(-e float_mult) set path = ( . \$path )
if(-e \`dirname \$0\`/float_mult ) set path = ( \`dirname \$0\` \$path )
rm -f sigma.map \${tempfile}output.bin
head -c \$head variance.map >! \${tempfile}temp.map
tail -c +\$skip variance.map >! \${tempfile}variance.bin
float_mult \${tempfile}variance.bin \${tempfile}variance.bin \${tempfile}output.bin -power1 0.5 -power2 0 > /dev/null
if(-e \${tempfile}output.bin) then
    cat \${tempfile}output.bin >> \${tempfile}temp.map
    echo "scale factor 1 0" | mapmask mapin \${tempfile}temp.map mapout sigma.map > /dev/null
    rm -f \${tempfile}output.bin
    echo "sigma.map is the rms deviation from \$ref"
else
    if(! -e float_mult.c) goto compile_float_mult
endif
rm -f \${tempfile}temp.map
rm -f \${tempfile}variance.bin
rm -f \${tempfile}output.bin


echo "sigma.map :"
echo "go" | mapdump mapin sigma.map | egrep density



exit

#############################################################################
#############################################################################


compile_float_mult:
echo "attempting to generate float_mult utility ..."
cat << EOF >! float_mult.c

/* multiply two binary "float" files together                                           -James Holton           1-31-10

example:

gcc -O -O -o float_mult float_mult.c -lm
./float_mult file1.bin file2.bin output.bin 

 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>


char *infile1name = "";
char *infile2name = "";
FILE *infile1 = NULL;
FILE *infile2 = NULL;
char *outfilename = "output.bin\\\\0";
FILE *outfile = NULL;

int main(int argc, char** argv)
{
     
    int n,i,j,k,pixels;
    float *outimage;
    float *inimage1;
    float *inimage2;
    float power1=1.0,power2=1.0;
    float sum,sumd,sumsq,sumdsq,avg,rms,rmsd,min,max;
        
    /* check argument list */
    for(i=1; i<argc; ++i)
    {
        if(strlen(argv[i]) > 4)
        {
            if(strstr(argv[i]+strlen(argv[i])-4,".bin"))
            {
                printf("filename: %s\\\\n",argv[i]);
                if(infile1 == NULL){
                    infile1 = fopen(argv[i],"r");
                    if(infile1 != NULL) infile1name = argv[i];
                }
                else
                {
                    if(infile2 == NULL){
                        infile2 = fopen(argv[i],"r");
                        if(infile2 != NULL) infile2name = argv[i];
                    }
                    else
                    {
                        outfilename = argv[i];
                    }
                }
            }
        }

        if(argv[i][0] == '-')
        {
            /* option specified */
            if(strstr(argv[i], "-power1") && (argc >= (i+1)))
            {
                power1 = atof(argv[i+1]);
            }
            if(strstr(argv[i], "-power2") && (argc >= (i+1)))
            {
                power2 = atof(argv[i+1]);
            }
        }
    }

    if(infile1 == NULL || infile2 == NULL){
        printf("usage: float_mult file1.bin file2.bin [outfile.bin] -power1 1.0 -power2 1.0\\\\n");
        printf("options:\\\\n");\\\\
//        printf("\\\\t-atom\\\\tnumber of atoms in the following file\\\\n");
//      printf("\\\\t-file filename.txt\\\\ttext file containing point scatterer coordinates in Angstrom relative to the origin.  The x axis is the x-ray beam and Y and Z are parallel to the detector Y and X coordinates, respectively\\\\n");
exit(9);
    }


    /* load first float-image */
    fseek(infile1,0,SEEK_END);
    n = ftell(infile1);
    rewind(infile1);
    inimage1 = calloc(n,1);
    inimage2 = calloc(n,1);
    fread(inimage1,n,1,infile1);
    fclose(infile1);
    fread(inimage2,n,1,infile2);
    fclose(infile2);

    pixels = n/sizeof(float);
    outfile = fopen(outfilename,"w");
    if(outfile == NULL)
    {
        printf("ERROR: unable to open %s\\\\n", outfilename);
        exit(9);
    }
    

    outimage = calloc(pixels,sizeof(float));
    sum = sumsq = sumd = sumdsq = 0.0;
    min = 1e99;max=-1e99;
    for(i=0;i<pixels;++i)
    {
        if(inimage1[i]<0.0 && power1 != ((int) power1) ) inimage1[i] = 0.0;
        if(inimage2[i]<0.0 && power2 != ((int) power2) ) inimage2[i] = 0.0;
        outimage[i] = powf(inimage1[i],power1) * powf(inimage2[i],power2);
        if(outimage[i]>max) max=outimage[i];
        if(outimage[i]<min) min=outimage[i];
        sum += outimage[i];
        sumsq += outimage[i]*outimage[i];
    }
    avg = sum/pixels;
    rms = sqrt(sumsq/pixels);
    for(i=0;i<pixels;++i)
    {
        sumd   += outimage[i] - avg;
        sumdsq += (outimage[i] - avg) * (outimage[i] - avg);
    }
    rmsd = sqrt(sumdsq/pixels);
    printf("max = %g min = %g\\\\n",max,min);
    printf("mean = %g rms = %g rmsd = %g\\\\n",avg,rms,rmsd);


    printf("writing %s as %d %d-byte floats\\\\n",outfilename,pixels,sizeof(float));
    outfile = fopen(outfilename,"w");
    fwrite(outimage,pixels,sizeof(float),outfile);
    fclose(outfile);


    return;
}

EOF
gcc -o float_mult float_mult.c -lm -static
set path = ( . \$path )
goto sqrt



EOF-script
chmod a+x map_rmsd.com


set path = ( . $path )
rehash
goto got_map_rmsd







######################################################################
#
#	notes and tests
#
#
set id = 1w3m

getcif.com ${id}
phenix.ready_set ${id}.pdb 
cad hklin1 ${id}.mtz hklout badscale.mtz << EOF
labin file 1 all
scale file 1 0.5
EOF
phenix.refine badscale.mtz ${id}.updated.pdb ${id}.metal.edits ${id}.link.edits \
 main.number_of_macro_cycles=0

~/projects/map_noise/RAPID_END.com ${id}.updated_refine_001.eff

