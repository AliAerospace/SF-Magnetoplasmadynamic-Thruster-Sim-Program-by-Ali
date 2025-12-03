function MPD_Sim_AliAerospace

fig = uifigure('Name','Self Field MPD Thruster Simulator by AliAerospace','Position',[100 100 1500 800]);

%CONTROL INPUTS
uilabel(fig,'Text','Electron Temp (eV):','Position',[30 720 180 22]);
TeMenu = uidropdown(fig,'Items',{'2','3','4','5'},'Value','4',...
    'Position',[210 720 80 22]);
uilabel(fig,'Text','','Position',[300 720 120 22],'FontAngle','italic');


uilabel(fig,'Text','Cathode Radius (m):','Position',[30 685 180 22]);
RcField = uieditfield(fig,'numeric','Position',[210 685 80 22],'Value',0.01);
uilabel(fig,'Text','Typical: 0.005–0.02 m','Position',[300 685 150 22],'FontAngle','italic');


uilabel(fig,'Text','Anode Radius (m):','Position',[30 650 180 22]);
RaField = uieditfield(fig,'numeric','Position',[210 650 80 22],'Value',0.05);
uilabel(fig,'Text','Typical: 0.03–0.08 m','Position',[300 650 150 22],'FontAngle','italic');


uilabel(fig,'Text','Axial Length (m):','Position',[30 615 180 22]);
LzField = uieditfield(fig,'numeric','Position',[210 615 80 22],'Value',0.1);
uilabel(fig,'Text','Typical: 0.05–0.3 m','Position',[300 615 150 22],'FontAngle','italic');


uilabel(fig,'Text','Mass Flux (kg/m²·s):','Position',[30 580 180 22]);
FluxField = uieditfield(fig,'numeric','Position',[210 580 80 22],'Value',0.7958);
uilabel(fig,'Text','Typical: 0.2–2','Position',[300 580 150 22],'FontAngle','italic');


uilabel(fig,'Text','Ionisation Cross-Section $S_{ion}$ (m²):','Interpreter','latex',...
    'Position',[30 545 220 22]);
SionField = uieditfield(fig,'numeric','Position',[260 545 100 22],'Value',2.8e-20);
uilabel(fig,'Text','','Position',[370 545 150 22],'FontAngle','italic');


uilabel(fig,'Text','e–Neutral Cross-Section $S_{en}$ (m²):','Interpreter','latex',...
    'Position',[30 510 220 22]);
SenField = uieditfield(fig,'numeric','Position',[260 510 100 22],'Value',15e-20);
uilabel(fig,'Text','','Position',[370 510 150 22],'FontAngle','italic');


uilabel(fig,'Text','Gas:','Position',[30 475 180 22]);
GasMenu = uidropdown(fig,'Items',{'Argon','Xenon','Hydrogen'},'Value','Argon',...
    'Position',[210 475 100 22]);


ResetButton = uibutton(fig,'push','Text','Reset (Argon Defaults)','FontWeight','bold',...
    'Position',[100 345 180 30],'ButtonPushedFcn',@resetDefaults);


uiimage(fig,'Position',[50 60 280 250],'ImageSource','mpd.png','ScaleMethod','fit');
uilabel(fig,'Text','Self-Field MPD Thruster','Position',[100 40 200 20],...
    'FontAngle','italic','FontSize',11);


RunButton = uibutton(fig,'push','Text','Run Simulation','FontWeight','bold',...
    'Position',[100 310 180 30],'ButtonPushedFcn',@RunSimulation);


ThrustLabel = uilabel(fig, 'Text', 'Thrust: -- N', ...
    'FontSize', 13, 'FontWeight', 'bold', 'FontColor', [0 0 0.4], ...
    'Position', [60 440 250 25]);

VelocityLabel = uilabel(fig, 'Text', 'Exit Velocity: -- m/s', ...
    'FontSize', 13, 'FontWeight', 'bold', 'FontColor', [0 0 0.4], ...
    'Position', [60 420 250 25]);

IspLabel = uilabel(fig, 'Text', 'Specific Impulse: -- s', ...
    'FontSize', 13, 'FontWeight', 'bold', 'FontColor', [0 0 0.4], ...
    'Position', [60 400 250 25]);

EtaLabel = uilabel(fig, 'Text', 'Efficiency: -- %', ...
    'FontSize', 13, 'FontWeight', 'bold', 'FontColor', [0 0.4 0], ...
    'Position', [60 380 250 25]);

tg = uitabgroup(fig,'Position',[420 30 1040 740]);
tab1 = uitab(tg,'Title','Full Solution');
tab2 = uitab(tg,'Title','Subsonic Region');
tab3 = uitab(tg,'Title','Collisional Rates');

axC = gobjects(6,1);
axS = gobjects(6,1);
for i=1:6
    col = mod(i-1,3);
    row = ceil(i/3);
    axC(i) = uiaxes(tab1,'Position',[60+320*col, 440-270*(row-1), 300, 250]);
    axS(i) = uiaxes(tab2,'Position',[60+320*col, 440-270*(row-1), 300, 250]);
end
axR = uiaxes(tab3,'Position',[150 100 750 500]);

%Callbacks
    function resetDefaults(~,~)
        GasMenu.Value = 'Argon';
        TeMenu.Value = '4';
        RcField.Value = 0.01;
        RaField.Value = 0.05;
        LzField.Value = 0.1;
        FluxField.Value = 0.7958;
        SionField.Value = 2.8e-20;
        SenField.Value  = 15e-20;
        uialert(fig,'Reset to Argon default parameters.','Defaults Applied');
    end

%SIMULATION
    function RunSimulation(~,~)
        % Read values
        T_em = str2double(TeMenu.Value);
        r_c  = RcField.Value;
        r_a  = RaField.Value;
        L_z  = LzField.Value;
        flux_mass = FluxField.Value;
        S_ion = SionField.Value;
        S_en  = SenField.Value;

        %SELF FIELD MAGNETOPLASMADYNAMIC THRUSTER SIMULATOR

        %CONSTANTS and PARAMETERS
        g        = 9.80665;              %gravitational acceleration (m/s^2)
        e        = 1.6021765e-19;        %elementary charge (C)
        m_e      = 9.11e-31;             %electron mass (kg)
        u_mass   = 1.660539e-27;         %atomic mass unit (kg)
        gas      = 39.948;               %atomic mass number (Argon) [CHOOSE]
        m_i      = gas * u_mass;         %ion mass (kg)
        mu_0     = pi * 4 * 1e-7;        %magnetic permeability of free space (H/m)
        epsilon_0 = 8.854e-12;           %permittivity of free space (F/m)
        T_e      = T_em * e;             %electron temperature (J)
        L_x      = r_a - r_c;            %radial gap (m)
        L_y      = pi * (r_a + r_c);     %azimuthal perimeter (m)
        flux_ref = flux_mass;            %reference mass flux
        T_ref     = T_e;                 %reference temperature (J)
        L_ref     = L_z;                 %reference length (m)

        switch GasMenu.Value
            case 'Argon'
                gas = 39.948;
                E_ion = 15.76 * e;  % eV → J
            case 'Xenon'
                gas = 131.29;
                E_ion = 12.13 * e;
            case 'Hydrogen'
                gas = 1.008;
                E_ion = 13.6 * e;
        end


        %DIMENSIONAL ANALYSIS AND NORMALISATION
        u_ref     = sqrt(T_ref / m_i);                   %reference velocity (m/s)
        n_ref     = flux_ref / (m_i * u_ref);            %reference density (m^-3)
        nu_ref    = u_ref / L_ref;                       %frequency scale (s^-1)
        Q_ref     = nu_ref / n_ref;                      %reference collisional coefficient (m^3/s)
        E_ref     = T_ref / (e * L_ref);                 %reference electric field (V/m)
        B_ref     = E_ref / u_ref;                       %reference magnetic field (T)
        sigma_ref = e^2 / (m_i * Q_ref);                 %reference conductivity (S/m)
        R_m       = sigma_ref * mu_0 * u_ref * L_ref;    %magnetic Reynolds number (–)
        K_2    = 10.5e-10;               %charge-exchange rate constant (m^3/s)
        K_1    = 1.67e-10;               %empirical constant for charge-exchange (m^3/s)
        ln_LAMBDA = 9;                   %Coulomb logarithm (–)

        %IONISATION AND COLLISIONAL RATE COEFFICIENTS (dimensional)
        Q_ion = sqrt((8*T_e)/(pi*m_e)) * S_ion * (1 + ((T_e*E_ion)/(T_e + E_ion)^2)) * exp(-E_ion/T_e);
        Q_en  = sqrt((8*T_e)/(pi*m_e)) * S_en;
        Q_cx  = u_ref * (K_2 - K_1*log10(u_ref))^2;
        Q_ei  = (T_e/e)^(-3/2) * ln_LAMBDA * 2.9e-12;

        %NON-DIMENSIONALIZED COLLISIONAL PARAMETERS
        q_ion = Q_ion / Q_ref;
        q_en  = (Q_en / Q_ref) * (m_e/m_i);
        q_cx  = Q_cx / Q_ref;
        q_ei  = (Q_ei / Q_ref) * (m_e/m_i);


        %%%%%%%%%%%%%%

        %INITIAL CONDITIONS OF SONIC POINTS BASED ON USER SELECTED T_e

        T_e_table = [2, 3, 4, 5];
        %lookup table
        zS_table  = [0, 0, 0, 0];
        uiS_table = [1, 1, 1, 1];
        giS_table = [0.7, 0.625, 0.87, 0.84];
        bS_table  = [13, 13.8, 11.9, 11.9];
        unS_table = [0.75, 0.66, 0.56245, 0.613];

        idx = find(T_e_table == T_em, 1);
        z_s = zS_table(idx);
        u_i_s = uiS_table(idx);
        g_i_s = giS_table(idx);
        b_s = bS_table(idx);
        u_n_s = unS_table(idx);

        %DERIVED VALUES AT SONIC POINT
        n_i_s   = g_i_s / u_i_s;                          %ion density (–)
        n_n_s   = (1 - g_i_s) / u_n_s;                    %neutral density (–)
        sigma_s = n_i_s / (q_en * n_n_s + q_ei * n_i_s);  %normalised conductivity
        s_w     = 0;                                      %source term (–)

        %NORMALIZED ELECTRIC FIELD AT SONIC POINT
        E_x_s = ((u_i_s * n_i_s * n_n_s * (u_i_s - u_n_s) * (q_cx + q_ion) + ...
            (n_i_s * n_n_s * q_ion - s_w)) / ...
            (u_i_s * sigma_s * b_s)) + u_i_s * b_s;

        %PARAMETER VECTOR FOR ODE SOLVER
        P_opt = [E_x_s q_ion q_en q_cx q_ei s_w R_m];


        %EIGENVALUE PROBLEM FOR INITIAL CONDITIONS SUB AND SUPER SONIC

        S0 = [z_s, u_i_s, n_i_s, b_s, u_n_s];      %state vector at sonic point
        S0n = [1, u_i_s, n_i_s, b_s, u_n_s];       %normalised

        delta = 0.001;                             %small perturbation magnitude
        DELTA = delta * S0n;                       %scaled perturbation vector

        %Generate perturbed states (forward and backward perturbations)
        YSupper = [z_s + DELTA(1), u_i_s, n_i_s, b_s, u_n_s;
            z_s, u_i_s + DELTA(2), n_i_s, b_s, u_n_s;
            z_s, u_i_s, n_i_s + DELTA(3), b_s, u_n_s;
            z_s, u_i_s, n_i_s, b_s + DELTA(4), u_n_s;
            z_s, u_i_s, n_i_s, b_s, u_n_s + DELTA(5)];

        YSlower = [z_s - DELTA(1), u_i_s, n_i_s, b_s, u_n_s;
            z_s, u_i_s - DELTA(2), n_i_s, b_s, u_n_s;
            z_s, u_i_s, n_i_s - DELTA(3), b_s, u_n_s;
            z_s, u_i_s, n_i_s, b_s - DELTA(4), u_n_s;
            z_s, u_i_s, n_i_s, b_s, u_n_s - DELTA(5)];

        for k=1:5
            fSupper(:,k) = M1(YSupper(k,:), P_opt);
            fSlower(:,k) = M1(YSlower(k,:), P_opt);
        end

        for r = 1:5
            for s = 1:5
                JacfyS(r,s) = (fSupper(r,s) - fSlower(r,s)) / (2 * DELTA(r));
            end
        end

        [V, D] = eig(JacfyS);
        [~, col] = find(D == max(max(D)));
        V1 = V(:, col);
        V2(3:5) = V1(3:5)' ./ S0(3:5);
        V2(1:2) = V1(1:2);
        delta = 0.05 / max(abs(V2));
        VA = delta * V1;
        S0_sub = S0 - sign(VA(1)) * VA';             %subsonic region start
        S0_sup = S0 + sign(VA(1)) * VA';             %supersonic region start


        %SS
        chi_span_sup = [0, 10000];
        options_sup = odeset('Events',@sup_limit,...
            'AbsTol',[1e-4 1e-4 1e-4 1e-4 1e-4]);

        P_opt_sup = [E_x_s q_ion q_en q_cx q_ei s_w R_m];

        [chi_sup, Y_sup] = ode45(@ODE_function, chi_span_sup, S0_sup, options_sup, P_opt_sup);
        z_sup   = Y_sup(:,1);       %axial coordinate (–)
        u_i_sup = Y_sup(:,2);       %ion velocity (–)
        n_i_sup = Y_sup(:,3);       %ion density (–)
        b_sup   = Y_sup(:,4);       %magnetic field (–)
        u_n_sup = Y_sup(:,5);       %neutral velocity (–)

        n_n_sup   = (1 - u_i_sup .* n_i_sup) ./ u_n_sup;                       %neutral density (–)
        sigma_sup = n_i_sup ./ (q_ei .* n_i_sup + q_en .* n_n_sup);            %conductivity (–)

        G_sup  = u_i_sup .* b_sup .* sigma_sup .* (E_x_s - u_i_sup .* b_sup) ...
            - n_i_sup .* n_n_sup .* q_ion;                                 %net source term (–)

        G_plus = u_i_sup .* b_sup .* sigma_sup .* (E_x_s - u_i_sup .* b_sup);   %positive contribution (EM)
        G_minus = -n_i_sup .* n_n_sup .* q_ion;                                %negative contribution (ionisation)

        db_sup = -n_i_sup .* (u_i_sup.^2 - 1) .* sigma_sup .* (E_x_s - u_i_sup .* b_sup); %dB/dχ (–)

        %Final values at end of domain
        z_final = z_sup(end);
        b_final = b_sup(end);
        G_final = G_sup(end);


        %SUBS
        chi_span_sub = [0, 10000];
        options_sub = odeset('Events',@sub_limit,...
            'AbsTol',[1e-4 1e-4 1e-4 1e-4 1e-4]);
        P_opt_sub = [E_x_s q_ion q_en q_cx q_ei s_w R_m];
        [chi_sub, Y_sub] = ode45(@ODE_function, chi_span_sub, S0_sub, options_sub, P_opt_sub);

        z_sub   = Y_sub(:,1);
        u_i_sub = Y_sub(:,2);
        n_i_sub = Y_sub(:,3);
        b_sub   = Y_sub(:,4);
        u_n_sub = Y_sub(:,5);

        n_n_sub = (1 - u_i_sub .* n_i_sub) ./ u_n_sub;


        %convert back
        n_sub = size(z_sub,1);
        n_sup = size(z_sup,1);

        z_total  = zeros(n_sub + n_sup,1);
        u_i_total = zeros(n_sub + n_sup,1);
        n_i_total = zeros(n_sub + n_sup,1);
        b_total   = zeros(n_sub + n_sup,1);
        u_n_total = zeros(n_sub + n_sup,1);

        for i = 1:n_sub
            z_total(i) = z_sub(n_sub + 1 - i);
        end
        for i = 1:n_sup
            z_total(i + n_sub) = z_sup(i);
        end

        z_total = z_total - z_sub(end);

        for i = 1:n_sub
            u_i_total(i) = u_i_sub(n_sub + 1 - i);
            n_i_total(i) = n_i_sub(n_sub + 1 - i);
            b_total(i)   = b_sub(n_sub + 1 - i);
            u_n_total(i) = u_n_sub(n_sub + 1 - i);
        end
        for i = 1:n_sup
            u_i_total(i + n_sub) = u_i_sup(i);
            n_i_total(i + n_sub) = n_i_sup(i);
            b_total(i + n_sub)   = b_sup(i);
            u_n_total(i + n_sub) = u_n_sup(i);
        end

        n_n_total = (1 - u_i_total .* n_i_total) ./ u_n_total;

        Z  = z_total * L_ref;               %axial coordinate (m)
        U_i = u_i_total * u_ref;            %ion velocity (m/s)
        N_i = n_i_total * n_ref;            %ion density (m^-3)
        B_dim = b_total * B_ref;            %magnetic field (T)
        U_n = u_n_total * u_ref;            %neutral velocity (m/s)
        N_n = (u_ref * n_ref - U_i .* N_i) ./ U_n;  %neutral density (m^-3)
        N_n_alt = n_n_total * n_ref;
        Error = N_n - N_n_alt;

        sigma_dim = sigma_ref * (n_i_total ./ ...
            (n_i_total * q_ei + (N_n / n_ref) * q_en));
        E_check = E_x_s / (u_i_total(end) * u_ref * b_total(end) * B_ref);
        u_effective = (U_i(end)*N_i(end) + U_n(end)*N_n(end)) / (N_i(end) + N_n(end));
        I_sp = u_effective / g;
        B_anode = B_dim(1);

        A_channel = L_x * L_y;             %channel area (m^2)
        L_z_obt = z_total(end) * L_ref;    %physical length (m)
        m_dot = m_i * (N_i(end) + N_n(end)) * u_effective * A_channel;  %mass flow rate (kg/s)

        F_thrust = u_effective * m_dot;

        P_useful = 0.5 * F_thrust * u_effective;

        V_d = E_x_s * L_x;
        I_d = B_anode * L_y / mu_0;

        P_discharge = V_d * I_d;

        ETA = P_useful / P_discharge;
        % ===== Update UI Outputs =====
        ThrustLabel.Text = sprintf('Thrust: %.3e N', F_thrust);
        VelocityLabel.Text = sprintf('Exit Velocity: %.2f m/s', u_effective);
        IspLabel.Text = sprintf('Specific Impulse: %.2f s', I_sp);
        EtaLabel.Text = sprintf('Efficiency: %.2f %%', ETA);

        n_0 = N_i(1);
        G_i_A  = u_i_total(1) * n_i_total(1);
        U_n_A  = U_n(1);
        u_i_A  = u_i_total(1);
        U_i_A  = U_i(1);
        G_i_E  = u_i_total(end) * n_i_total(end);
        Z_sub  = z_sub * L_ref;
        U_i_sub_dim = u_i_sub * u_ref;
        N_i_sub_dim = n_i_sub * n_ref;
        B_sub_dim   = b_sub * B_ref;
        U_n_sub_dim = u_n_sub * u_ref;
        N_n_sub_dim = (1 - u_i_sub .* n_i_sub) ./ u_n_sub * n_ref;
        G_i_sub = N_i_sub_dim .* U_i_sub_dim;
        G_n_sub = N_n_sub_dim .* U_n_sub_dim;
        G_i_dim = N_i .* U_i;
        G_n_dim = N_n .* U_n;



        function dY = M1(Y, P_opt)
            z   = Y(1);    %normalised position
            u_i = Y(2);    %ion velocity
            n_i = Y(3);    %ion density
            b   = Y(4);    %magnetic field
            u_n = Y(5);    %neutral velocity

            E_x   = P_opt(1);   %normalised electric field
            q_ion = P_opt(2);   %ionisation collision rate
            q_en  = P_opt(3);   %electron-neutral collision rate
            q_cx  = P_opt(4);   %charge-exchange rate
            q_ei  = P_opt(5);   %electron-ion collision rate
            s_w   = P_opt(6);   %source term
            R_m   = P_opt(7);   %magnetic Reynolds number
            g_i   = n_i * u_i;                     %ion flux (dimensionless)
            n_n   = (1 - g_i) / u_n;               %neutral density (dimensionless)
            sigma = n_i / (n_i * q_ei + n_n * q_en); %normalised conductivity

            %ODEs these are non-dimensional!
            dz   = n_i * (u_i^2 - 1);
            du_i = u_i * sigma * b * (E_x - u_i * b) ...
                - u_i * n_i * n_n * (u_i - u_n) * (q_cx + q_ion) ...
                - (n_i * n_n * q_ion - s_w);
            dn_i = -n_i * sigma * b * (E_x - u_i * b) ...
                + n_i^2 * n_n * (u_i - u_n) * (q_cx + q_ion) ...
                + (n_i * u_i) * (n_i * n_n * q_ion - s_w);
            db   = -dz * sigma * R_m * (E_x - u_i * b);
            du_n = dz * (n_i * n_n * (u_i - u_n) * q_cx - u_n * s_w) / (1 - g_i);

            dY = [dz; du_i; dn_i; db; du_n];
        end

        function [value, c1, d1] = sup_limit(~, S, P_opt)
            value(1) = S(4);
            value(2) = S(1) - 100;
            c1 = [1, 1];
            d1 = [0, 0];
        end

        function [value, c2, d2] = sub_limit(~, S, P_opt)
            value(1) = S(2) + 1;
            value(2) = S(1) - 10;
            value(3) = S(5) - 1e-6;
            c2 = [1, 1, 1];
            d2 = [0, 0, 0];
        end

        function dS = ODE_function(~, S, P_opt)
            %Solves the coupled plasma eqns
            E_x  = P_opt(1);      %dimensionless electric field
            q_ion = P_opt(2);     %ionisation collision frequency ratio
            q_en  = P_opt(3);     %electron-neutral collision ratio
            q_cx  = P_opt(4);     %charge exchange collision ratio
            q_ei  = P_opt(5);     %electron-ion collision ratio
            s_w   = P_opt(6);     %ionisation source term
            R_m   = P_opt(7);     %magnetic Reynolds number

            z   = S(1);           %axial position (–)
            u_i = S(2);           %ion velocity (–)
            n_i = S(3);           %ion density (–)
            b   = S(4);           %magnetic field (–)
            u_n = S(5);           %neutral velocity (–)

            g_i = n_i * u_i;               %ion flux (–)
            n_n = (1 - g_i) / u_n;         %neutral density (–)
            sigma = n_i / (n_i*q_ei + n_n*q_en);   %dimensionless conductivity

            %Differential equations (from conservation and Maxwell's laws)
            dz   = n_i * (u_i^2 - 1);                                                 %axial scaling
            du_i = u_i*sigma*b*(E_x - u_i*b) ...                                      %Lorentz + electric term
                - u_i*n_i*n_n*(u_i - u_n)*(q_cx + q_ion) ...                         %momentum exchange
                - (n_i*n_n*q_ion - s_w);                                             %source/sink
            dn_i = -n_i*sigma*b*(E_x - u_i*b) ...                                     %ion continuity
                + n_i^2*n_n*(u_i - u_n)*(q_cx + q_ion) ...
                + (n_i*u_i)*(n_i*n_n*q_ion - s_w);
            db   = -dz*sigma*R_m*(E_x - u_i*b);                                       %induction equation
            du_n = dz*(n_i*n_n*(u_i - u_n)*q_cx - u_n*s_w) / (1 - g_i);               %neutral momentum

            %Return derivative vector
            dS = [dz; du_i; dn_i; db; du_n];

        end

        cla(axC(:));
        labels = {...
            {'Ion flux, $N_i U_i$ [m$^{-2}$s$^{-1}$]', G_i_dim},...
            {'Ion velocity, $U_i$ [m/s]', U_i},...
            {'Ion density, $N_i$ [m$^{-3}$]', N_i},...
            {'Magnetic field, $B$ [T]', B_dim},...
            {'Neutral flux, $N_n U_n$ [m$^{-2}$s$^{-1}$]', G_n_dim},...
            {'Neutral velocity, $U_n$ [m/s]', U_n}};
        for i=1:6
            plot(axC(i),Z,labels{i}{2},'k','LineWidth',2);
            xlabel(axC(i),'Axial position, Z [m]','Interpreter','latex');
            ylabel(axC(i),labels{i}{1},'Interpreter','latex');
            grid(axC(i),'on');
        end

        cla(axS(:));
        subLabels = {...
            {'Ion density, $N_i$ [m$^{-3}$]', N_i_sub_dim},...
            {'Ion velocity, $U_i$ [m/s]', U_i_sub_dim},...
            {'Ion flux, $G_i$ [m$^{-2}$s$^{-1}$]', G_i_sub},...
            {'Neutral velocity, $U_n$ [m/s]', U_n_sub_dim},...
            {'Magnetic field, $B$ [T]', B_sub_dim},...
            {'Neutral flux, $G_n$ [m$^{-2}$s$^{-1}$]', G_n_sub}};
        for i=1:6
            plot(axS(i),Z_sub,subLabels{i}{2},'k','LineWidth',2);
            xlabel(axS(i),'Axial position, Z [m]','Interpreter','latex');
            ylabel(axS(i),subLabels{i}{1},'Interpreter','latex');
            grid(axS(i),'on');
        end

        cla(axR);
        e = 1.6021765e-19; m_e = 9.11e-31; E_ion = 15.75 * e;
        Te_range = linspace(1,20,100);
        Qion = zeros(size(Te_range)); Qen=Qion; Qcx=Qion; Qei=Qion;
        for i = 1:length(Te_range)
            T_e_i = Te_range(i) * e;
            Qion(i) = sqrt((8*T_e_i)/(pi*m_e)) * S_ion * ...
                (1 + ((T_e_i*E_ion)/(T_e_i + E_ion)^2)) * exp(-E_ion/T_e_i);
            Qen(i)  = sqrt((8*T_e_i)/(pi*m_e)) * S_en;
            Qcx(i)  = u_ref * (K_2 - K_1*log10(u_ref))^2;
            Qei(i)  = (T_e_i/e)^(-3/2) * ln_LAMBDA * 2.9e-12;
        end
        semilogy(axR,Te_range,Qion,'k-','LineWidth',2); hold(axR,'on');
        semilogy(axR,Te_range,Qen,'k--','LineWidth',2);
        semilogy(axR,Te_range,Qcx,'k-.','LineWidth',2);
        semilogy(axR,Te_range,Qei,'k:','LineWidth',2);
        xlabel(axR,'Electron Temperature $T_e$ [eV]','Interpreter','latex');
        ylabel(axR,'Rate Coefficient $Q$ [m$^3$/s]','Interpreter','latex');
        legend(axR,{...
            '$Q_{\text{ionisation}}$', ...
            '$Q_{\text{electron-neutral}}$', ...
            '$Q_{\text{charge-exchange}}$', ...
            '$Q_{\text{electron-ion}}$'}, ...
            'Interpreter','latex','Location','northeast');
        grid(axR,'on');
        title(axR,'Collisional Rate Coefficients vs $T_e$','Interpreter','latex');
        xlim(axR,[2 5]);

        uialert(fig,'Simulation completed successfully!','MPD Simulator');
    end
end



