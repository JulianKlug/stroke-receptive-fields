import os, argparse
import subprocess


def mcflirt(infile, reffile=None, out_prefix='mcf', outdir=None, verbose=False, stages=3,
            sinc_final=False, spline_final=False):
    '''
    MCFLIRT is an intra-modal motion correction tool designed for use on fMRI time series and based on optimization and
    registration techniques used in FLIRT, a fully automated robust and accurate tool for linear (affine) inter- and
    inter-modal brain image registration.

    Note: an official implementation exists in the fslpy package, its implementation is however not well documented (at the time of writing)
    Ref: Jenkinson, M., Bannister, P., Brady, J. M. and Smith, S. M. Improved Optimisation for the Robust and Accurate Linear Registration and Motion Correction of Brain Images. NeuroImage, 17(2), 825-841, 2002.
    :param infile: path to the file to apply motion correction to
    :param refvol: reference volume for motion correction, default: mean volume
    :param out_prefix: prefix for transformed file, default: mcf
    :param outdir: directory for output, default: input directory
    :param verbose: default: false
    :param stages: default is 3. 4 specifies final (internal) sinc interpolation
    If a 4-stage correction has been specified, a further optimization pass is carried out using sinc
    interpolation (internally) for greater accuracy. This step is significantly more time-consuming than the previous
    part of the correction, which should take in the order of 10 minutes for a 100 volume timeseries. This internal
    interpolation is independent of the final resampling interpolation (as specified by sinc_final or spline_final).
    :return: outfile: path to output file
    '''

    indir, infile = os.path.split(infile)
    if outdir is None:
        outdir = indir
    outfile = os.path.join(outdir, out_prefix + '_' + infile)

    mcflirt_cli = ['mcflirt',
                   '-in', infile,
                   '-out', outfile,
                   '-stages', str(stages)]

    if reffile is not None:
        mcflirt_cli += ['-reffile', reffile]
    if verbose:
        mcflirt_cli += ['-report']
    if sinc_final:
        mcflirt_cli += ['-sinc_final']
    if spline_final:
        mcflirt_cli += ['-spline_final']

    subprocess.run(mcflirt_cli, cwd=indir)
    return outfile

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Python wrapper for mcflirt: motion correction')
    parser.add_argument('input_directory')
    args = parser.parse_args()
    mcflirt(args.input_directory)