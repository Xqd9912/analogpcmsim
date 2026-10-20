# Dry-run stub for run_twostage_job.sh flow testing on the login node: NO VASP is executed.
# Defines a fake `mpirun` that writes minimal OSZICAR/OUTCAR/WAVECAR/CHGCAR into the cwd.
# Fake CHGCARs are copied from T2TS_FAKE_SRC (a prior T2 run dir with default/ and atomaug_prep/).
mpirun() {
  local ich=$(awk -F= '/^ICHARG/{gsub(/ /,"",$2);print $2}' INCAR)
  printf "       N       E                     dE\nDAV:   1    -0.1E+02\nRMM:   2    -0.2E+02\nRMM:   3    -0.2E+02\n   1 F= -.2E+02 E0= -.2E+02\n" > OSZICAR
  { echo "   ISTART =      $(awk -F= '/^ISTART/{gsub(/ /,"",$2);print $2}' INCAR)"; echo "   ICHARG =      $ich";
    echo "   dimension x,y,z NGX =    30 NGY =   30 NGZ =   40"; echo "   dimension x,y,z NGXF=    60 NGYF=   60 NGZF=   80"
    echo "   NELECT =      29.0000"; echo "  free  energy   TOTEN  =       -20.0 eV"
    echo "------------------------ aborting loop because EDIFF is reached ---"; echo " General timing and accounting"
    echo "                            Elapsed time (sec):        1.000"; } > OUTCAR
  echo "fake vasp: ICHARG=$ich CHGCAR_in=$( [ -s CHGCAR ] && head -1 CHGCAR || echo none) WAVECAR_in=$( [ -s WAVECAR ] && cat WAVECAR || echo none)"
  grep -q "^LWAVE = True" INCAR && echo "wavecar-from-$(basename $(dirname $PWD))/$(basename $PWD)" > WAVECAR
  if grep -q "^LCHARG = True" INCAR; then
    case $ich in 2|0) cp $T2TS_FAKE_SRC/default/CHGCAR CHGCAR;; 12) cp $T2TS_FAKE_SRC/atomaug_prep/CHGCAR CHGCAR;; esac
  fi
}
