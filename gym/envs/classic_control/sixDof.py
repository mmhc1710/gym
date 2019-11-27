# -*- coding: utf-8 -*-
"""
Created on Wed Oct 25 12:32:22 2017

@author: daksh
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import numpy as np
import scipy.constants as cons

def aircraft(X_0, control):
    
    VT      = X_0[0]
    alpha   = X_0[1]
    beta    = X_0[2]
    	
    phi     = X_0[3]
    theta   = X_0[4]
    psi     = X_0[5]

    P       = X_0[6]
    Q       = X_0[7]
    R       = X_0[8]
    
    # Initial Control surfaces
    # del_t_x   = X_0[12]
    # del_e_x   = X_0[13]
    # del_a_x   = X_0[14]
    # del_r_x   = X_0[15]
    
    # Control inputs
    del_t   = control[0]
    del_e   = control[1]
    del_a   = control[2]
    del_r   = control[3]
    	
    u = VT*np.cos(alpha)*np.cos(beta)
    v = VT*np.sin(beta)
    w = VT*np.sin(alpha)*np.cos(beta)
    
    # Constants
    g = cons.g*3.2808399
    	
    # Trim values
    del_etrim    = 0.0261799387799149 #[rad]
    VTtrim       = 50.6343 #[ft/s]
    alphatrim    = 0.0127409035395586 #[rad]
    Ptrim        = 0 #[rad/s]
    Qtrim        = 0 #[rad/s]
    Rtrim        = 0 #[rad/s]
    utrim        = VTtrim*np.cos(alphatrim) #[ft/s]
    
    #Geometry
    S    = 4.819060764 #[ft2]
    cbar = 8.776004922/12 #[ft]
    b    = 82.5/12 #[ft]
    AR   = 9.808057486
    
    #Mass and Inertia
    mass = 8.27 #[lbm]
    I_xx = 0.1 #[slug-ft2]
    I_yy = 0.1 #[slug-ft2]
    I_zz = 0.35 #[slug-ft2]
    I_xz = 0.00 #[slug-ft2]
    
    #Steady state coefficients
    C_L1  = 0.586132204721337
    C_m1  = 0.005202371113864
    C_mt1 = -0.005202371113863
    
    #Stability and control derivatives
    C_D0bar   = 0.06
    C_yb      = -0.555693945810164
    C_yp      = -0.206459374251021
    C_yr      = 0.275012701278722
    C_yda     = 0
    C_ydr     = 0.228226346074142
    C_La      = 5.777445465319050
    C_Lq      = 6.337008667888560
    C_Ladot   = 1.315462351274260
    C_Lu      = 0.001215412337750
    C_Ldele   = 0.293178615611292
    C_lb      = -0.161510951944297
    C_lp      = -0.547061903960014
    C_lr      = 0.166319742584319
    C_ldela   = 0.334940367071198
    C_ldelr   = 0.017542815458796
    C_ma      = -0.888816147836906
    C_mq      = -15.232971677527400
    C_madot   = -4.606431389471500
    C_mu      = 1.893711150768030e-04
    C_mdele   = -1.15
    C_mtu     = 0.016867461874563
    C_mtalpha = -0.086032474442540
    C_nb      = 0.123565567748892
    C_np      = -0.069311640205122
    C_nr      = -0.114473208365030
    C_ndela   = 0
    C_ndelr   = -0.094902203003894
    
    #Thrust modelling constants
    xT2 = 0.2327*4.7358/(100**2)
    xT1 = 0.2327*0.7428/100
    xT0 = 0.2327*-0.0521
    dT  = 3/12
    
    #Constants
    mass = mass/g # lbm to slug
    rho  = 0.002295
    qbar = 0.5*rho*(VT**2)
    
    #Aerodynamic forces and moment coefficients
    C_L  = C_L1 + C_La*(alpha - alphatrim) + (C_Lq*(Q - Qtrim)*(cbar/2))/utrim + (C_Lu*(u - utrim))/utrim + C_Ldele*(del_e - del_etrim)# + (C_Ladot*alphadot*(cbar/2))/utrim
    C_D  = C_D0bar + (C_L**2)/(np.pi*AR*0.88)
    C_Y  = C_yb*beta + (C_yp*(P - Ptrim)*(b/2))/utrim + (C_yr*(R - Rtrim)*(b/2))/utrim + C_yda*del_a + C_ydr*del_r
    C_ls = C_lb*beta + (C_lp*(P - Ptrim)*(b/2))/utrim + (C_lr*(R - Rtrim)*(b/2))/utrim + C_ldela*del_a + C_ldelr*del_r
    C_ms = C_m1 + C_ma*(alpha - alphatrim) + (C_mq*(Q - Qtrim)*(cbar/2))/utrim + (C_mu*(u - utrim))/utrim + C_mdele*(del_e - del_etrim) + 2*(C_m1*(u - utrim))/utrim + (C_mtu + 2*C_mt1)*(u - utrim)/utrim + C_mtalpha*(alpha - alphatrim)# + (C_madot*alphadot*(cbar/2))/utrim
    C_ns = C_nb*beta + (C_np*(P - Ptrim)*(b/2))/utrim + (C_nr*(R - Rtrim)*(b/2))/utrim + C_ndela*(del_a) + C_ndelr*del_r
    C_xa = C_L*np.sin(alpha) - C_D*np.cos(alpha)
    C_ya = C_Y
    C_za = -C_L*np.cos(alpha) - C_D*np.sin(alpha)
    #print(C_Ldele*(del_e - del_etrim))# = (qbar*S*C_za)/mass
    C_l  = C_ls*np.cos(alpha) - C_ns*np.sin(alpha)
    C_m  = C_ms
    C_n  = C_ls*np.sin(alpha) + C_ns*np.cos(alpha)
    
    #Thrust model
    T = xT0 + xT1*del_t*100 + xT2*((del_t*100)**2)
    
    #Gravitational acceleration
    g_x = -g*np.sin(theta)
    g_y = g*np.sin(phi)*np.cos(theta)
    g_z = g*np.cos(phi)*np.cos(theta)
    
    #Body acceleration
    a_x = (qbar*S*C_xa + T)/mass
    a_y = (qbar*S*C_ya)/mass
    a_z = (qbar*S*C_za)/mass
    	
    #Moments
    L = C_l*qbar*S*b
    M = C_m*qbar*S*cbar - T*dT
    N = C_n*qbar*S*b
    	
    ## Differential equations:
    	
    # Body linear velocities
    udot = a_x + g_x + R*v - Q*w
    vdot = a_y + g_y - R*u + P*w
    wdot = a_z + g_z + Q*u - P*v
    	
    # Total velocity and airflow angles
    VTdot    = (u*udot + v*vdot + w*wdot)/VT
    alphadot = (u*wdot - w*udot)/((u**2) + (w**2))
    #betadot  = ((VT*vdot - v*VTdot)/((u**2) + (w**2)))*np.cos(beta)
    betadot  = (vdot*((u**2) + (w**2)) - v*(u*udot + w*wdot))/((VT**2)*(np.sqrt((u**2) + (w**2))))
    
    # Body angular velocities
    Pdot = (I_zz*L + I_xz*N - (I_xz*(I_yy - I_xx - I_zz)*P + ((I_xz**2) + I_zz*(I_zz - I_yy))*R)*Q)/(I_xx*I_zz - (I_xz**2))
    Qdot = (M - (I_xx - I_zz)*P*R - I_xz*((P**2) - (R**2)))/I_yy
    Rdot = (I_xz*L + I_xx*N + (I_xz*(I_yy - I_xx - I_zz)*R + ((I_xz**2) + I_xx*(I_xx - I_yy))*P)*Q)/(I_xx*I_zz - (I_xz**2))
    	
    # Inertial (Euler) angles
    phidot   = P + (R*np.cos(phi) + Q*np.sin(phi))*np.tan(theta)
    thetadot = Q*np.cos(phi) - R*np.sin(phi)
    psidot   = (Q*np.sin(phi) + R*np.cos(phi))/np.cos(theta)
    
    # Inertial position dynamics
    # x_Idot = (np.cos(theta)*np.cos(psi))*u + (-np.cos(phi)*np.sin(psi) + np.sin(phi)*np.sin(theta)*np.cos(psi))*v + (np.sin(phi)*np.sin(psi) + np.cos(phi)*np.sin(theta)*np.cos(psi))*w
    # y_Idot = (np.cos(theta)*np.sin(psi))*u + (np.sin(phi)*np.sin(theta)*np.sin(psi) + np.cos(phi)*np.cos(psi))*v + (np.cos(phi)*np.sin(theta)*np.sin(psi) - np.sin(phi)*np.cos(psi))*w
    # z_Idot = (-np.sin(theta))*u + (np.sin(phi)*np.cos(theta))*v + (np.cos(phi)*np.cos(theta))*w
    
    # Servo dynamics
    # Athrottle=-40
    # Bthrottle=40
    # Aelevator=-35
    # Belevator=35
    # Aaileron=-30
    # Baileron=30
    # Arudder=-50
    # Brudder=50
    
    # del_tdot = Athrottle*del_t_x + Bthrottle*del_t
    # del_edot = Aelevator*del_e_x + Belevator*del_e
    # del_adot = Aaileron*del_a_x + Baileron*del_a
    # del_rdot = Arudder*del_r_x + Brudder*del_r
    
    Xdot = np.array([VTdot, alphadot, betadot, phidot, thetadot, psidot, Pdot, Qdot, Rdot])#, x_Idot, y_Idot, z_Idot])
    # Xdot = np.array([VTdot, alphadot, betadot, phidot, thetadot, psidot, Pdot, Qdot, Rdot, x_Idot, y_Idot, z_Idot, del_tdot, del_edot, del_adot, del_rdot])
    
    return Xdot


# def control_dynamics(X_0, control):
#
#     # Initial Control surfaces
#     del_t_x = X_0[0]
#     del_e_x = X_0[1]
#     del_a_x = X_0[2]
#     del_r_x = X_0[3]
#
#     # Control inputs
#     del_t   = control[0]
#     del_e   = control[1]
#     del_a   = control[2]
#     del_r   = control[3]
#
#     # Servo dynamics
#     Athrottle=-40
#     Bthrottle=40
#     Aelevator=-35
#     Belevator=35
#     Aaileron=-30
#     Baileron=30
#     Arudder=-50
#     Brudder=50
#
#     del_tdot = Athrottle*del_t_x + Bthrottle*del_t
#     del_edot = Aelevator*del_e_x + Belevator*del_e
#     del_adot = Aaileron*del_a_x + Baileron*del_a
#     del_rdot = Arudder*del_r_x + Brudder*del_r
#
#     Xdot = np.array([del_tdot, del_edot, del_adot, del_rdot])
#
#     return Xdot

# def alphadot_fun(X_0, alphadot, control):
#
#     VT      = X_0[0]
#     alpha   = X_0[1]
#     beta    = X_0[2]
#
#     phi     = X_0[3]
#     theta   = X_0[4]
#
#     P       = X_0[6]
#     Q       = X_0[7]
#     R       = X_0[8]
#
#     # Control inputs
#     del_t   = control[0]
#     del_e   = control[1]
#
#     u = VT*np.cos(alpha)*np.cos(beta)
#     v = VT*np.sin(beta)
#     w = VT*np.sin(alpha)*np.cos(beta)
#
#     # Constants
#     g = cons.g*3.2808399
#
#     # Trim values
#     del_etrim    = 0.0261799387799149 #[rad]
#     VTtrim       = 50.6343 #[ft/s]
#     alphatrim    = 0.0127409035395586 #[rad]
#     Qtrim        = 0 #[rad/s]
#     utrim        = VTtrim*np.cos(alphatrim) #[ft/s]
#
#     #Geometry
#     S    = 4.819060764 #[ft2]
#     cbar = 8.776004922/12 #[ft]
#     AR   = 9.808057486
#
#     #Mass and Inertia
#     mass = 8.27 #[lbm]
#
#     #Steady state coefficients
#     C_L1  = 0.586132204721337
#
#     #Stability and control derivatives
#     C_D0bar   = 0.06
#     C_La      = 5.777445465319050
#     C_Lq      = 6.337008667888560
#     C_Ladot   = 1.315462351274260
#     C_Lu      = 0.001215412337750
#     C_Ldele   = 0.293178615611292
#
#     #Thrust modelling constants
#     xT2 = 0.2327*4.7358/np.square(100)
#     xT1 = 0.2327*0.7428/100
#     xT0 = 0.2327*-0.0521
#
#     #Constants
#     mass = mass/g # lbm to slug
#     rho  = 0.002295
#     qbar = 0.5*rho*(VT**2)
#
#     #Aerodynamic forces and moment coefficients
#     C_L  = C_L1 + C_La*(alpha - alphatrim) + (C_Lq*(Q - Qtrim)*(cbar/2))/utrim + (C_Ladot*alphadot*(cbar/2))/utrim + (C_Lu*(u - utrim))/utrim + C_Ldele*(del_e - del_etrim)
#     C_D  = C_D0bar + (C_L**2)/(np.pi*AR*0.88)
#     C_xa = C_L*np.sin(alpha) - C_D*np.cos(alpha)
#     C_za = -C_L*np.cos(alpha) - C_D*np.sin(alpha)
#
#     #Thrust model
#     T = xT0 + xT1*del_t*100 + xT2*((del_t*100)**2)
#
#     #Gravitational acceleration
#     g_x = -g*np.sin(theta)
#     g_z = g*np.cos(phi)*np.cos(theta)
#
#     #Body acceleration
#     a_x = (qbar*S*C_xa + T)/mass
#     a_z = (qbar*S*C_za)/mass
#
#     # Body linear velocities
#     udot = a_x + g_x + R*v - Q*w
#     wdot = a_z + g_z + Q*u - P*v
#
#     # Total velocity and airflow angles
#     alphadot = (u*wdot - w*udot)/((u**2) + (w**2))
#
#     return alphadot

def rk4_states(X_0, dt, control):
    
    dt2 = dt/2
    dt6 = dt/6
    
    dxdt = aircraft(X_0, control)#, alphadot, control)
    x0t = X_0 + dt2*dxdt
    
    dx0t = aircraft(x0t, control)#, alphadot, control)
    x0t = X_0 + dt2*dx0t
    
    dxmold = aircraft(x0t, control)#, alphadot, control)
    x0t = X_0 + dt*dxmold
    
    dxm = dx0t + dxmold
    
    dx0t = aircraft(x0t, control)#, alphadot, control)
    statesdot = dxdt + dx0t + 2*dxm
    X = X_0 + dt6*statesdot
    
    psi_deg = X[5]*(180/np.pi)
    if (psi_deg < -180) or (psi_deg > 180):
        psi_deg = (psi_deg + 180) % 360
        if psi_deg == 0:
            psi_deg = 360
        psi_deg = psi_deg - 180
    X[5] = psi_deg*(np.pi/180)
    
    # control = X_0[15-3:15]
    # alphadot = alphadot_fun(X_0, alphadot, control)

    return X
    # return X, alphadot


# def rk4_actions(X_0, dt, control):
#     dt2 = dt / 2
#     dt6 = dt / 6
#
#     dxdt = control_dynamics(X_0, control)
#     x0t = X_0 + dt2 * dxdt
#
#     dx0t = control_dynamics(x0t, control)
#     x0t = X_0 + dt2 * dx0t
#
#     dxmold = control_dynamics(x0t, control)
#     x0t = X_0 + dt * dxmold
#
#     dxm = dx0t + dxmold
#
#     dx0t = control_dynamics(x0t, control)
#     statesdot = dxdt + dx0t + 2 * dxm
#     X = X_0 + dt6 * statesdot
#
#     return X
