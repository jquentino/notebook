#!/usr/bin/env python-sirius
"""Script for running LOCO algorithm."""
import pickle as _pickle
import numpy as np
import time

import siriuspy.clientconfigdb as servconf
from pymodels import si
import pyaccel as pyac
from apsuite.loco.config import LOCOConfigSI
from apsuite.loco.main import LOCO
from apsuite.optics_analysis.tune_correction import TuneCorr
import sys
import apsuite.commisslib as commisslib


def save_data(fname,
              config,
              model,
              gbpm, gcorr, rbpm, rdip,
              chi_history, energy_shift, residue_history,
              girder_shift, kldelta_history, ksldelta_history):
    """."""
    data = dict(fit_model=model,
                config=config,
                gain_bpm=gbpm,
                gain_corr=gcorr,
                roll_bpm=rbpm,
                roll_dip=rdip,
                energy_shift=energy_shift,
                chi_history=chi_history,
                res_history=residue_history,
                girder_shift=girder_shift,
                kldelta_history=kldelta_history,
                ksldelta_history=ksldelta_history)
    if not fname.endswith('.pickle'):
        fname += '.pickle'
    with open(fname, 'wb') as fil:
        _pickle.dump(data, fil)


def load_data(fname):
    """."""
    sys.modules['apsuite.commissioning_scripts'] = commisslib
    if not fname.endswith('.pickle'):
        fname += '.pickle'
    with open(fname, 'rb') as fil:
        data = _pickle.load(fil)
    return data


def move_tunes(model, loco_setup):
    """."""
    tunex_goal = 49 + loco_setup['tunex']
    tuney_goal = 14 + loco_setup['tuney']

    print('--- correcting si tunes...')
    tunecorr = TuneCorr(
        model, 'SI', method='Proportional', grouping='TwoKnobs')
    tunecorr.get_tunes(model)
    print('    tunes init  : ', tunecorr.get_tunes(model))
    tunemat = tunecorr.calc_jacobian_matrix()
    tunecorr.correct_parameters(
           model=model,
           goal_parameters=np.array([tunex_goal, tuney_goal]),
           jacobian_matrix=tunemat, tol=1e-10)
    print('    tunes final : ', tunecorr.get_tunes(model))


def create_loco_config(loco_setup, change_tunes=True):
    """."""
    config = LOCOConfigSI()

    # create nominal model
    model = si.create_accelerator()

    # dimension used in the fitting
    config.dim = '6d'

    # change nominal tunes to match the measured values
    if change_tunes:
        move_tunes(model, loco_setup)

    config.model = model

    # initial gains (None set all gains to one and roll to zero)
    config.gain_bpm = None
    config.gain_corr = None
    config.roll_bpm = None

    # # # load previous fitting
    # folder = ''
    # fname = folder + 'fitting_after_b1b2corr'
    # data = load_data(fname)
    # config.model = data['fit_model']
    # config.gain_bpm = data['gain_bpm']
    # config.gain_corr = data['gain_corr']
    # model.cavity_on = True
    # model.radiation_on = True

    if config.dim == '4d':
        config.model.cavity_on = False
        config.model.radiation_on = False
    elif config.dim == '6d':
        config.model.cavity_on = True
        config.model.radiation_on = False

    # Select if LOCO includes dispersion column in matrix, diagonal and
    # off-diagonal elements
    config.use_dispersion = True
    config.use_diagonal = True
    config.use_offdiagonal = True

    # Set if want to fit quadrupoles and dipoles in families instead of
    # individually
    config.use_quad_families = False
    config.use_dip_families = False

    # Add constraints in gradients
    config.constraint_deltak_step = True
    config.constraint_deltak_total = False
    config.deltakl_normalization = 1e-3

    config.tolerance_delta = 1e-6
    config.tolerance_overfit = 1e-6

    # Jacobian Inversion method, LevenbergMarquardt requires transpose method

    config.inv_method = LOCOConfigSI.INVERSION.Transpose
    # config.inv_method = LOCOConfigSI.INVERSION.Normal

    # config.min_method = LOCOConfigSI.MINIMIZATION.GaussNewton
    # config.lambda_lm = 0

    config.min_method = LOCOConfigSI.MINIMIZATION.LevenbergMarquardt
    config.lambda_lm = 1e-3  # NOTE: Default: 1e-3
    config.fixed_lambda = False

    # quadrupolar strengths to be included in the fit
    config.fit_quadrupoles = True
    config.fit_sextupoles = False
    config.fit_dipoles = False

    # Select subset of families to be fit, 'None' will include all families by
    # default
    config.quadrupoles_to_fit = None
    config.sextupoles_to_fit = None
    config.skew_quadrupoles_to_fit = config.famname_skewquadset.copy()
    fc2_idx = config.skew_quadrupoles_to_fit.index('FC2')
    config.skew_quadrupoles_to_fit.pop(fc2_idx)
    # config.skew_quadrupoles_to_fit = ['SFA0', 'SDB0', 'SDP0']
    config.dipoles_to_fit = None
    config.update()

    # dipolar errors at dipoles
    config.fit_dipoles_kick = False

    # off diagonal elements fitting
    if config.use_offdiagonal:
        # To correct the coupling, set just config.fit_skew_quadrupoles and
        # fit_roll_bpm to True and the others to False
        config.fit_quadrupoles_coupling = False
        config.fit_sextupoles_coupling = False
        config.fit_dipoles_coupling = False

        config.fit_roll_bpm = True
        config.fit_skew_quadrupoles = True
    else:
        config.fit_quadrupoles_coupling = False
        config.fit_sextupoles_coupling = False
        config.fit_dipoles_coupling = False
        config.fit_roll_bpm = False
        config.fit_skew_quadrupoles = False

    config.fit_energy_shift = False

    # BPM and corrector gains (always True by default)
    config.fit_gain_bpm = False
    config.fit_gain_corr = True

    # kicks used in the measurements
    config.delta_kickx_meas = 15e-6  # [rad]
    config.delta_kicky_meas = 1.5 * 15e-6  # [rad]
    config.delta_frequency_meas = 15 * 5  # [Hz]

    # girders shifts
    config.fit_girder_shift = False

    # dipoles roll errors
    config.fit_dip_roll = False

    # initial weights

    # BPMs
    config.weight_bpm = None
    # config.weight_bpm = 1/loco_setup['bpm_variation'].flatten()

    # Correctors
    # config.weight_corr = None
    config.weight_corr = np.ones(281)
    # Weight on dispersion column, 280 to be as important as the other columns
    # and 1e6 to match the order of magnitudes. The dispersion factor can be
    # set to force the dispersion fitting harder.
    dispersion_factor = 2
    config.weight_corr[-1] = dispersion_factor * (75/15 * 1e6)

    # Gradients constraints
    # config.weight_deltak = None

    # Remember Quad_number = 270, Sext_number = 280, Dip_number = 100
    config.weight_deltak = np.ones(270 + 0*280 + 0*100)

    # # singular values selection method
    # config.svd_method = LOCOConfigSI.SVD.Threshold
    # config.svd_thre = 1e-6
    config.svd_method = LOCOConfigSI.SVD.Selection

    # # When adding the gradient constraint, it is required to remove only the
    # # last singular value
    # # Remember Quad Number = 270, BPM gains = 2 * 160, BPM roll = 1 * 160,
    # # Corrector Gains = 120 + 160, Dip Number = 100, Sext Number = 280,
    # # QS Number = 80
    # config.svd_sel = 270 + 2 * 160 + (120 + 160) + 0 * 280 + 0 * 100 + 80 - 1

    # One can simplify this by setting config.svd_sel = -1, but the way above
    # is far more explict
    config.svd_sel = -1
    config.parallel = True
    return config


def create_loco(
        loco_setup,
        load_jacobian=False, save_jacobian=False,
        change_tunes=True, roll_dip=False):
    """."""
    config = create_loco_config(loco_setup, change_tunes=change_tunes)
    config.fit_dip_roll = roll_dip
    client = servconf.ConfigDBClient(config_type='si_orbcorr_respm')
    orbmat_meas = np.array(
        client.get_config_value(name=loco_setup['orbmat_name']))
    orbmat_meas = np.reshape(orbmat_meas, (320, 281))
    orbmat_meas[:, -1] *= 1e-6  # convert dispersion column from um to m
    config.goalmat = orbmat_meas

    # swap BPM for test
    # config.goalmat[[26,25], :] = config.goalmat[[25, 26], :]
    # config.goalmat[[160+26, 160+25], :] = config.goalmat[[160+25, 160+26], :]

    # swap CH for test
    # nfig.goalmat[:, [24, 23]] = config.goalmat[:, [23, 24]]

    # swap CV for test
    # config.goalmat[:, [120+32, 120+31]] = config.goalmat[:, [120+31, 120+32]]

    alpha = pyac.optics.get_mcf(config.model)
    rf_frequency = loco_setup['rf_frequency']
    config.measured_dispersion = -1 * alpha * rf_frequency * orbmat_meas[:, -1]

    config.update()
    print('')
    print(config)

    print('[create loco object]')
    loco = LOCO(config=config, save_jacobian_matrices=save_jacobian)

    kl_folder = 'jacobian_KL/' + config.dim
    ksl_folder = 'jacobian_KsL/' + config.dim

    if load_jacobian:
        # Pre-calculated jacobian can be used (remember that these matrices
        # were calculated at nominal tunes nux=49.096 and nuy=14.152)

        loco.update(
            fname_jloco_k_dip=kl_folder+'/dipoles',
            fname_jloco_k_quad=kl_folder+'/quadrupoles',
            fname_jloco_k_sext=kl_folder+'/sextupoles',
            fname_jloco_ks_dip=ksl_folder+'/dipoles',
            fname_jloco_ks_quad=ksl_folder+'/quadrupoles',
            fname_jloco_ks_sext=ksl_folder+'/sextupoles',
            fname_jloco_ks_skewquad=ksl_folder+'/skew_quadrupoles')

        # # The other option is to read the recent calculated jacobian at
        # # appropriated tunes. LOCO class saves the jacobian files with
        # # default names, please rename it in the folder and use it here

        # loco.update(
        #    fname_jloco_k_quad='jloco_quadrupole_name',
        #    fname_jloco_ks_skewquad='jloco_skewquadrupole_name')
    else:
        loco.update()
        # loco.update(fname_jloco_k_quad='6d_KL_quadrupoles_trims',
        #        fname_jloco_ks_skewquad='6d_KsL_skew_quadrupoles',)
    return loco


def run_and_save(
        setup_name, file_name, niter,
        load_jacobian=False, save_jacobian=False,
        change_tunes=True, roll_dip=False):
    """."""
    setup = load_data(setup_name)
    if 'data' in setup.keys():
        setup = setup['data']

    t0 = time.time()
    loco = create_loco(
        setup,
        load_jacobian=load_jacobian, save_jacobian=save_jacobian,
        change_tunes=change_tunes, roll_dip=roll_dip)
    loco.run_fit(niter=niter)
    save_data(
        fname=file_name,
        config=loco.config,
        model=loco.fitmodel,
        gbpm=loco.bpm_gain,
        gcorr=loco.corr_gain,
        rbpm=loco.bpm_roll,
        rdip=loco.dip_roll,
        energy_shift=loco.energy_shift,
        chi_history=loco.chi_history,
        residue_history=loco.residue_history,
        girder_shift=loco.girder_shift,
        kldelta_history=loco.kldelta_history,
        ksldelta_history=loco.ksldelta_history)
    dt = time.time() - t0
    print('running time: {:.1f} minutes'.format(dt/60))
    return loco


# run_and_save(
#     setup_name='loco_input_ac_orm_loco_iter3',
#     file_name='fitting_ac_orm_loco_iter3',
#     niter=20,
#     change_tunes=True,
#     load_jacobian=False,
#     save_jacobian=False)
