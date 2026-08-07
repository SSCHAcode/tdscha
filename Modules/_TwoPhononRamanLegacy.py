"""Private source archive for the disabled two-phonon Raman implementation.

This module is deliberately not installed or imported.  The former implementation
is stored as inert text for audit/reference only; it must not be executed.  The
same source is available at commit b4f6dc2af70796b0bc42323b9333f017bacdc0a0.
"""

LEGACY_SOURCE_COMMIT = "b4f6dc2af70796b0bc42323b9333f017bacdc0a0"

LEGACY_TWO_PHONON_RAMAN_SOURCE = r'''    def prepare_unpolarized_raman_FT(self, index = 0, debug = False, eq_raman_tns = None, use_symm = True,\
                                     ens_av_raman = None, raman_tns_ens = None, add_2ph = True):
        """
        PREPARE UNPOLARIZED RAMAN SIGNAL CONSIDERING FLUCTUATIONS OF THE RAMAN TENSOR
        =============================================================================

        The raman tensor is read from the dynamical matrix provided by the original ensemble.

        The perturbations are prepared accordin to the formula (see https://doi.org/10.1021/jp5125266)

        ..math:

            I_unpol = 45/9 (xx + yy + zz)^2
                      + 7/2 [(xx-yy)^2 + (xx-zz)^2 + (yy-zz)^2]
                      + 7 * 3 [(xy)^2 + (yz)^2 + (xz)^2]

        Parameters:
        -----------
            -index: the pol component of the unpolarized signal
            -debug: if true we save the second order Raman tensor
            -eq_raman_tns: np.array with shape (3, 3, 3 * N_at_uc), the equilibirum raman tensor
            -use_symm: bool, if True symmetries are enforced
            -ens_av_raman:  the ensemble on which we compute the averages of the Raman tensors
            -raman_tns_ens: np.array with shape (N_conf, 3, 3, 3 * N_at_sc), the raman tensors on the displaced configruations
        """
        # Check if the raman tensor is present
        assert not self.dyn.raman_tensor is None, "Error, no Raman tensor found. Cannot initialize the Raman responce"

        labels = [i for i in range(7)]
        if not(index in labels):
            raise ValueError('{} should be in {}'.format(index, labels))

        epols = {'x' : np.array([1,0,0]),\
                 'y' : np.array([0,1,0]),\
                 'z' : np.array([0,0,1])}

        # (xx + yy + zz)^2
        if index == 0:
            # raman_v  = self.dyn.GetRamanVector(epols['x'], epols['x'])
            # raman_v += self.dyn.GetRamanVector(epols['y'], epols['y'])
            # raman_v += self.dyn.GetRamanVector(epols['z'], epols['z'])
            self.prepare_anharmonic_raman_FT(raman = raman_tns_ens, raman_eq = eq_raman_tns,\
                                             pol_in   = epols['x'], pol_out   = epols['x'],\
                                             mixed = True,\
                                             pol_in_2 = epols['y'], pol_out_2 = epols['y'],\
                                             pol_in_3 = epols['z'], pol_out_3 = epols['z'],\
                                             add_two_ph = add_2ph, symmetrize = use_symm,\
                                             ensemble = ens_av_raman,\
                                             save_raman_tensor2 = debug, file_raman_tensor2 = 'xx_plus_yy_plus_zz')
        # (xx - yy)^2
        elif index == 1:
            # raman_v  = self.dyn.GetRamanVector(epols['x'], epols['x'])
            # raman_v -= self.dyn.GetRamanVector(epols['y'], epols['y'])
            # NB we put just one minus sign because the component is (xx - yy)^2
            self.prepare_anharmonic_raman_FT(raman = raman_tns_ens, raman_eq = eq_raman_tns,\
                                             pol_in   =  epols['x'],  pol_out  =  epols['x'],\
                                             mixed = True,\
                                             pol_in_2 = -epols['y'], pol_out_2 =  epols['y'],\
                                             pol_in_3 = np.zeros(3), pol_out_3 = np.zeros(3),\
                                             add_two_ph = add_2ph, symmetrize = use_symm,\
                                             ensemble = ens_av_raman,\
                                             save_raman_tensor2 = debug, file_raman_tensor2 = 'xx_minus_yy')
        # (xx - zz)^2
        elif index == 2:
            # raman_v  = self.dyn.GetRamanVector(epols['x'], epols['x'])
            # raman_v -= self.dyn.GetRamanVector(epols['z'], epols['z'])
            self.prepare_anharmonic_raman_FT(raman = raman_tns_ens, raman_eq = eq_raman_tns,\
                                             pol_in   =  epols['x'],  pol_out  =  epols['x'],\
                                             mixed = True,\
                                             pol_in_2 = -epols['z'], pol_out_2 =  epols['z'],\
                                             pol_in_3 = np.zeros(3), pol_out_3 = np.zeros(3),\
                                             add_two_ph = add_2ph, symmetrize = use_symm,\
                                             ensemble = ens_av_raman,\
                                             save_raman_tensor2 = debug, file_raman_tensor2 = 'xx_minus_zz')
        # (yy - zz)^2
        elif index == 3:
            # raman_v  = self.dyn.GetRamanVector(epols['y'], epols['y'])
            # raman_v -= self.dyn.GetRamanVector(epols['z'], epols['z'])
            self.prepare_anharmonic_raman_FT(raman = raman_tns_ens, raman_eq = eq_raman_tns,\
                                             pol_in   =  epols['y'],  pol_out  =  epols['y'],\
                                             mixed = True,\
                                             pol_in_2 = -epols['z'], pol_out_2 =  epols['z'],\
                                             pol_in_3 = np.zeros(3), pol_out_3 = np.zeros(3),\
                                             add_two_ph = add_2ph, symmetrize = use_symm,\
                                             ensemble = ens_av_raman,\
                                             save_raman_tensor2 = debug, file_raman_tensor2 = 'yy_minus_zz')
        # (xy)^2
        elif index == 4:
            # raman_v = self.dyn.GetRamanVector(epols['x'], epols['y'])
            self.prepare_anharmonic_raman_FT(raman = raman_tns_ens, raman_eq = eq_raman_tns,\
                                             pol_in   =  epols['x'],  pol_out  =  epols['y'],\
                                             mixed = False,\
                                             add_two_ph = add_2ph, symmetrize = use_symm,\
                                             ensemble = ens_av_raman,\
                                             save_raman_tensor2 = debug, file_raman_tensor2 = 'xy_square')
        # (xz)^2
        elif index == 5:
            # raman_v = self.dyn.GetRamanVector(epols['x'], epols['z'])
            self.prepare_anharmonic_raman_FT(raman = raman_tns_ens, raman_eq = eq_raman_tns,\
                                             pol_in   =  epols['x'],  pol_out  =  epols['z'],\
                                             mixed = False,\
                                             add_two_ph = add_2ph, symmetrize = use_symm,\
                                             ensemble = ens_av_raman,\
                                             save_raman_tensor2 = debug, file_raman_tensor2 = 'xz_square')
        # (yz)^2
        elif index == 6:
            # raman_v = self.dyn.GetRamanVector(epols['y'], epols['z'])
            self.prepare_anharmonic_raman_FT(raman = raman_tns_ens, raman_eq = eq_raman_tns,\
                                             pol_in   =  epols['y'],  pol_out  =  epols['z'],\
                                             mixed = False,\
                                             add_two_ph = add_2ph, symmetrize = use_symm,\
                                             ensemble = ens_av_raman,\
                                             save_raman_tensor2 = debug, file_raman_tensor2 = 'yz_square')

        return



    def prepare_anharmonic_raman_FT(self, raman = None, raman_eq = None,\
                                    pol_in = np.array([1.,0.,0.]), pol_out = np.array([1.,0.,0.]),\
                                    mixed = False, pol_in_2 = None, pol_out_2 = None,\
                                    pol_in_3 = None, pol_out_3 = None,\
                                    add_two_ph = False, symmetrize = False, ensemble = None,\
                                    save_raman_tensor2 = False, file_raman_tensor2 = None):
        r"""
        PREPARE THE PSI VECTOR FOR ANHARMONIC RAMAN SPECTRUM CALCULATION (NEW VERSION)
        ===========================================================================

        This works only with the Wigner representation if we add the two phonons effect.
        Prepare the psi vector for RAMAN spectrum considering position-dependent raman tensors.

        Parameters:
        -----------
            -raman: nd.array (N_configs, E_comp, E_comp, 3 * N_at_sc),
                 the Raman tensor for all configurations.
                 Indices are: Number of configuration, electric field component,
                 electric field component, atomic coordinates in sc.
            rama_eq: nd.array, (E_comp, E_comp, 3 * N_at_uc), the effective charges at equilibrium.
                 Indices are: electric field component,
                 electric field component, atomic coordinate in uc.
            -pol_in: nd.array, the polarization of in-out light. default is x
            -pol_out: nd.array, the polarization of in-out light. default is x
            -mixed: if True we can study the one and two phonon response to
                    pol_in \cdto \Xi \cdot pol_in + pol_in_2 \cdto \Xi \cdot pol_in_2 + pol_in_3 \cdto \Xi \cdot pol_in_3
                    (\Xi is the Raman tensor)
            -pol_in_2:  nd.array, the polarization of in-out light. default is None
            -pol_out_2: nd.array, the polarization of in-out light. default is None
            -pol_in_3:  nd.array, the polarization of in-out light. default is None
            -pol_out_3: nd.array, the polarization of in-out light. default is None
            -add_two_ph: bool, if True two phonon processes are included in the calculation
            -symmetrize: bool, if True the first/second order Raman tensors are symmetrized
            -ensemble: a scha ensemble object for computing the averages
            -save_raman_tensor2: bool if True we save the second order Raman tensor
        """
        if not self.use_wigner and add_two_ph:
            raise NotImplementedError('The two phonon processes are implemented only in Wigner')

        if raman is None:
            raise ValueError('Must specify the raman tensors for all configurations!')

        if mixed:
            #Check that we have the other polarization vectors
            if (pol_in_2 is None) or (pol_out_2 is None):
                raise ValueError('Must specify pol_in_2 pol_out_2 if mixed = True!')

            if (pol_in_3 is None) or (pol_out_3 is None):
                raise ValueError('Must specify pol_in_3 pol_out_3 if mixed = True!')

            if len(pol_in_2) != 3 or len(pol_out_2) != 3:
                raise ValueError('pol_in_2 pol_out_2 must be array of len 3')

            if len(pol_in_3) != 3 or len(pol_out_3) != 3:
                raise ValueError('pol_in_3 pol_out_3 must be array of len 3')


        print()
        print('PREPARE THE RAMAN ANHARMONIC SPECTRUM CALCULATION')
        print('=================================================')
        print('Are we considering two ph effects? = {}'.format(add_two_ph))
        print('Are we using Wigner? = {}'.format(self.use_wigner))
        print('Are we symmetrizing the raman tensor? = {}'.format(symmetrize))
        print()
        if ensemble is not None:
            Nconf = ensemble.N
        else:
            Nconf = self.N

        required = 'N_conf - E_field - E_field - 3 * N_at_sc'
        assert raman.shape[0] == Nconf, 'The raman tensor in input have the wrong shape. The required is {}'.format(required)
        assert raman.shape[1] == 3, 'The raman tensor in input have the wrong shape. The required is {}'.format(required)
        assert raman.shape[2] == 3, 'The raman tensor in input have the wrong shape. The required is {}'.format(required)
        assert raman.shape[3] == self.nat * 3, 'The raman tensor in input have the wrong shape. The required is {}'.format(required)

        # alpha is the polarizability

        # Get the average of the raman tensor, np.array with shape = (3, 3, 3 * N_at_sc)
        d1alpha_dR_av = perturbations.get_d1alpha_dR_av(ensemble, raman, symmetrize = symmetrize)

        # Get the supercell dyn then set the raman tensor euqal to d1alpha_dR_av
        sc_dyn = self.dyn.GenerateSupercellDyn(self.dyn.GetSupercell())
        sc_dyn.raman_tensor = d1alpha_dR_av

        # Get the Raman vector np.array (3 * N_at_sc)
        raman_vector_sc = sc_dyn.GetRamanVector(pol_in, pol_out)

        if mixed:
            print('ONE PH SECTOR adding compoent pol_in_2 pol_out_2 of the Raman tensor')
            raman_vector_sc += sc_dyn.GetRamanVector(pol_in_2, pol_out_2)
            print('ONE PH SECTOR adding compoent pol_in_3 pol_out_3 of the Raman tensor')
            raman_vector_sc += sc_dyn.GetRamanVector(pol_in_3, pol_out_3)


        # Now rescale by the mass and go in polarizaiton basis
        self.prepare_perturbation(raman_vector_sc, masses_exp = -1)
        print('[NEW] Pertubation modulus with one ph effects only = {}'.format(self.perturbation_modulus))
        print()

        # NOW PREPARE THE SECOND RAMAN TENSOR
        if add_two_ph:
            if raman_eq is not None:
                print('[NEW] Getting the equilibirum RAMAN tensor...')
                print()
                n_supercell = np.prod(self.dyn.GetSupercell())
                # raman_eq is np.array with shape = (E_field, E_field, N_at_uc * 3)
                raman_eq_size = np.shape(raman_eq)
                MSG = """
                Error, raman tns of the wrong shape: {}
                """.format(raman_eq_size)
                assert len(raman_eq_size) == 3, MSG
                if not self.ignore_small_w:
                    assert raman_eq_size[2] * n_supercell == self.nat * 3 #self.n_modes + 3
                assert raman_eq_size[0] == raman_eq_size[1] == 3

                # Get the raman tensor in the supercell (E_field, E_filed, 3 * N_at_sc)
                raman_eq_gamma = np.zeros((3, 3, 3 * n_supercell * self.dyn.structure.N_atoms), dtype = type(raman_eq[0,0,0]))
                raman_eq_gamma = np.tile(raman_eq, n_supercell)

            print('[NEW] Getting the two phonon contribution in RAMAN...')

            # d2M_dR np.array with shape = (3 * N_atoms, 3 * N_atoms, Efield)
            if raman_eq is not None:
                print('[NEW] Subtracting the equilibirum RAMAN tensor...')
                # raman - raman_eq_gamma, np.array with shape = (N_configs, Efield, Efield, 3 * N_at_sc)
                # THE RESULT HAS shape = (Efield, Efield, 3 * N_at_sc, 3 * N_at_sc)
                d2alpha_dR = perturbations.get_d2alpha_dR_av(ensemble, raman - raman_eq_gamma, None, symmetrize = symmetrize)
            else:
                # THE RESULT HAS shape = (Efield, Efield, 3 * N_at_sc, 3 * N_at_sc)
                d2alpha_dR = perturbations.get_d2alpha_dR_av(ensemble, raman, None, symmetrize = symmetrize)

            print('[NEW] Divide by the masses')
            # Divide by the masses of the atoms in the supercell shape =  (Efield, Efield, 3 * N_at_sc, 3 * N_at_sc)
            d2alpha_dR = np.einsum('c, abcd, d -> abcd', np.sqrt(self.m)**-1, d2alpha_dR, np.sqrt(self.m)**-1)

            if save_raman_tensor2:
                print('[NEW] Saving the second-order SCHA Raman tensor')
                np.save('{}'.format(file_raman_tensor2), d2alpha_dR)
                return

            print('[NEW] Go in polarization basis')
            # Now go in polarization basis, np.array with shape = (E_field, E_field, n_modes, n_modes)
            # d2alpha_dR_muspace = np.einsum('cm, abcd, dn -> abmn', self.pols, d2alpha_dR, self.pols)
            # -> substitute
            tmp                = np.einsum('abcd, cm -> abmd', d2alpha_dR, self.pols)
            d2alpha_dR_muspace = np.einsum('abmd, dn -> abmn', tmp, self.pols)

            # Project along the direction of the filed, np.array with shape = (n_modes, n_modes)
            dXi_dR_muspace = np.einsum('abmn, a, b -> mn', d2alpha_dR_muspace, pol_in, pol_out)

            if mixed:
                print('TWO PH SECTOR adding component pol_in_2 pol_out_2 of the Raman tensor')
                dXi_dR_muspace += np.einsum('abmn, a, b -> mn', d2alpha_dR_muspace, pol_in_2, pol_out_2)
                print('TWO PH SECTOR adding component pol_in_3 pol_out_3 of the Raman tensor')
                dXi_dR_muspace += np.einsum('abmn, a, b -> mn', d2alpha_dR_muspace, pol_in_3, pol_out_3)

            # Symmetrize in mu space, np.array with shape = (n_modes, n_modes)
            dXi_dR_muspace = 0.5 * (dXi_dR_muspace + dXi_dR_muspace.T)

            # Get chi_minus and chi_plus tensors, np.array with shape = (n_modes, n_modes)
            chi_minus = self.get_chi_minus()
            chi_plus  = self.get_chi_plus()

            # Get the pertubations on a'^(1) b'^(1)
            pert_a = -np.einsum('nm, nm -> nm', np.sqrt(-0.5 * chi_minus), dXi_dR_muspace)
            pert_b = +np.einsum('nm, nm -> nm', np.sqrt(+0.5 * chi_plus) , dXi_dR_muspace)

            # Check if everything is symmetric
            assert np.all(np.abs(dXi_dR_muspace - dXi_dR_muspace.T) < 1e-10), "Second derivative of the polarizability is not symmetric in pol basis"
            assert np.all(np.abs(pert_a - pert_a.T) < 1e-10), "a'(1) pertubation is not symmetric in pol basis"
            assert np.all(np.abs(pert_b - pert_b.T) < 1e-10), "b'(1) pertubation is not symmetric in pol basis"

            # Now get the perturbation for a'^(1)
            current = self.n_modes
            for i in range(self.n_modes):
                self.psi[current : current + self.n_modes - i] = pert_a[i, i:]
                current = current + self.n_modes - i

            # Now get the pertrubation for b'^(1)
            for i in range(self.n_modes):
                self.psi[current : current + self.n_modes - i] = pert_b[i, i:]
                current = current + self.n_modes - i

            # Add the mask dot taking into account symmetric elements
            mask_dot = self.mask_dot_wigner()
            # OVERWRITE the pertubation modulus considering the two phonon sector
            self.perturbation_modulus = self.psi.dot(self.psi * mask_dot)

            print('[NEW] Perturbation modulus after adding two ph contributions RAMAN = {}'.format(self.perturbation_modulus))
            print()

        return



    def prepare_anharmonic_raman_FT_2ph(self, d2alpha_dR = None, pol_in = np.array([1.,0.,0.]), pol_out = np.array([1.,0.,0.]),\
                                    mixed = False, pol_in_2 = None, pol_out_2 = None):
        r"""
        PREPARE THE PSI VECTOR FOR RAMAN SPECTRUM CALCULATION (NEW VERSION) DIRECTLY FROM 2nd ORDER RAMAN TENSOR
        ========================================================================================================

        This function is useful if we want to interpolate the 2nd Raman tensor on a bigger supercell.

        This works only with the Wigner representation if we add the two phonons effect.
        Prepare the psi vector for RAMAN spectrum considering position-dependent raman tensors.

        NOTE: we completely neglect the frist order Raman scattering!

        Parameters:
        -----------
            -d2alpha_dR: nd.array (E_comp, E_comp, 3 * N_at_sc, 3 * N_at_sc),
                 2nd order Raman tensor.
                 Indices are: Number of configuration, electric field component,
                 electric field component, atomic coordinates in sc.
            -pol_in: nd.array, the polarization of in-out light. default is x
            -pol_out: nd.array, the polarization of in-out light. default is x
            -mixed: if True we can study the one and two phonon response to
                    pol_in \cdot \Xi \cdot pol_out + pol_in_2 \cdot \Xi \cdot pol_out_2
                    (\Xi is the Raman tensor)
            -pol_in_2: nd.array, the polarization of in-out light. default is x
            -pol_out_2: nd.array, the polarization of in-out light. default is x
        """
        if not self.use_wigner:
            raise NotImplementedError('The two phonon processes are implemented only in Wigner')

        if d2alpha_dR is None:
            raise ValueError('Must specify the 2nd order Raman tensor!')

        exp_shape = (3, 3, self.nat * 3, self.nat * 3)
        if d2alpha_dR.shape != exp_shape:
            raise ValueError('The shape of the 2nd order Raman tensor is not correct, expected {}'.format(exp_shape))

        if mixed:
            if (pol_in_2 is None) or (pol_out_2 is None):
                raise ValueError('Must specify pol_in_2 pol_out_2 if mixed = True!')

            if len(pol_in_2) != 3 or len(pol_out_2) != 3:
                raise ValueError('pol_in_2 pol_out_2 must be array of len 3')


        print()
        print('PREPARE THE RAMAN ANHARMONIC SPECTRUM CALCULATION FROM 2nd ORDER RAMAN TENSOR')
        print('=============================================================================')
        # print('Are we considering two ph effects? = {}'.format(add_two_ph))
        print('Are we using Wigner? = {}'.format(self.use_wigner))
        # print('Are we symmetrizing the raman tensor? = {}'.format(symmetrize))
        print()

        print('TWO PH Going in polarization basis')
        # Now go in polarization basis, np.array with shape = (E_field, E_field, n_modes, n_modes)
        # d2alpha_dR_muspace = np.einsum('cm, abcd, dn -> abmn', self.pols, d2alpha_dR, self.pols)
        # -> substitute
        tmp                = np.einsum('abcd, cm -> abmd', d2alpha_dR, self.pols)
        d2alpha_dR_muspace = np.einsum('abmd, dn -> abmn', tmp, self.pols)
        # print(d2alpha_dR_muspace.shape)

        print('TWO PH Selecting the polarizations')
        # Project along the direction of the filed, np.array with shape = (n_modes, n_modes)
        dXi_dR_muspace = np.einsum('abmn, a, b -> mn', d2alpha_dR_muspace, pol_in, pol_out)
        # print(dXi_dR_muspace.shape)

        if mixed:
            print('TWO PH SECTOR adding component pol_in_2 pol_out_2 of the Raman tensor')
            dXi_dR_muspace += np.einsum('abmn, a, b -> mn', d2alpha_dR_muspace, pol_in_2, pol_out_2)

        # Symmetrize in mu space, np.array with shape = (n_modes, n_modes)
        dXi_dR_muspace = 0.5 * (dXi_dR_muspace + dXi_dR_muspace.T)

        # Get chi_minus and chi_plus tensors, np.array with shape = (n_modes, n_modes)
        chi_minus = self.get_chi_minus()
        chi_plus  = self.get_chi_plus()

        # Get the pertubations on a'^(1) b'^(1)
        pert_a = -np.einsum('nm, nm -> nm', np.sqrt(-0.5 * chi_minus), dXi_dR_muspace)
        pert_b = +np.einsum('nm, nm -> nm', np.sqrt(+0.5 * chi_plus) , dXi_dR_muspace)

        # Check if everything is symmetric
        assert np.all(np.abs(dXi_dR_muspace - dXi_dR_muspace.T) < 1e-10), "Second derivative of the polarizability is not symmetric in pol basis"
        assert np.all(np.abs(pert_a - pert_a.T) < 1e-10), "a'(1) pertubation is not symmetric in pol basis"
        assert np.all(np.abs(pert_b - pert_b.T) < 1e-10), "b'(1) pertubation is not symmetric in pol basis"

        print('[NEW] Perturbation modulus = {}'.format(self.perturbation_modulus))
        print()

        # Now get the perturbation for a'^(1)
        current = self.n_modes
        for i in range(self.n_modes):
            self.psi[current : current + self.n_modes - i] = pert_a[i, i:]
            current = current + self.n_modes - i

        # Now get the pertrubation for b'^(1)
        for i in range(self.n_modes):
            self.psi[current : current + self.n_modes - i] = pert_b[i, i:]
            current = current + self.n_modes - i

        # Add the mask dot taking into account symmetric elements
        mask_dot = self.mask_dot_wigner()
        # OVERWRITE the pertubation modulus considering the two phonon sector
        self.perturbation_modulus = self.psi.dot(self.psi * mask_dot)

        print('[NEW] Perturbation modulus adding two ph contributions RAMAN = {}'.format(self.perturbation_modulus))
        print()

        return



'''
