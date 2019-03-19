#!/usr/bin/env python
try:
    import argparse
    import sys
    import os
    import numpy as np
    import pandas as pd
    import scipy.interpolate
except BaseException as e:
    print('Ошибка импорта: ' + str(e))
    exit(1)

parser = argparse.ArgumentParser('BEST-2019[1]',
        description='Программа вычисляет координаты и направление сброса груза',
        epilog='\nNo licence')

parser.add_argument('-F', action='store',
        help='Файл с данными о аэродинамической силе (F.csv)',
        default='F.csv', type=str)
parser.add_argument('-W', '--Wind', action='store',
        help='Файл с данными о ветре (Wind.csv)',
        default='Wind.csv', type=str)
parser.add_argument('-H', '--height', action='store',
        help='Начальная высота (1400)', default=1400., type=float)
parser.add_argument('-S', '--speed', action='store',
        help='Начальная скорость (240)', default=250., type=float)
parser.add_argument('-M', '--mass', action='store',
        help='Масса тела (100)', default=100., type=float)
parser.add_argument('-P', '--picture', action='store',
        help='Сохранить изображение', type=str)
parser.add_argument('-O', '--output', action='store',
        help='Файл с траекторией и скоростями (result.csv)',
        type=str, default='result.csv')
parser.add_argument('-v', '--verbose', action='store',
        help='Выводит больше информации', default=1,
        type=int, nargs='?', const=1)

args = parser.parse_args()

def verbose_print(message, level=1, **kwargs):
    if args.verbose >= level:
        print(message, **kwargs)


try:
    F = pd.read_csv(args.F)
    Wind = pd.read_csv(args.Wind)
except BaseException as e:
    print("Ошибка открытия файла: " + str(e))
    exit(2)

g = -9.81
h = args.height
v_0 = args.speed
m = args.mass

def simple_quadratic_solve(a, b, c):
    D = b*b - 4*a*c
    if D < 0:
        return 0
    return (-b - D ** (1/2)) / a / 2

def calc_angle(v):
    return 180 / np.pi * np.arccos(v.dot(np.array([1, 0, 0])) / (v**2).sum() ** (1/2))

def find_coefs(X, y):
    return np.linalg.pinv(X.T.dot(X)).dot(X.T).dot(y)

def calc_regr(w, x):
    return x.dot(w.T)

def calc_Fa(w, x):
    return calc_regr(w, np.vstack([x, x**2]).T)

def calc_V(w, x):
    return calc_regr(w, np.vstack([x, x**(1/2)]).T)


interp_x = scipy.interpolate.interpolate.interp1d(Wind.Y, Wind.Wx, 'cubic')
interp_z = scipy.interpolate.interpolate.interp1d(Wind.Y, Wind.Wz, 'cubic')
interp_F = scipy.interpolate.interpolate.interp1d(F.V, F.Fa, 'cubic')
F_w = find_coefs(np.vstack([F.V, F.V**2]).T, F.Fa)
V_w = find_coefs(np.vstack([F.Fa, F.Fa**(1/2)]).T, F.V)

def vec_len(vec):
    return (np.sum(vec ** 2, 1) ** (1/2)).reshape(-1, 1)

def aero_acc(v_vel):
    vel = vec_len(v_vel)
    if vel == 0:
        vel = 1
    return v_vel / vel * (calc_Fa(F_w, vel) / m)

def Wind_vel(y):
    if y > 1400:
        y = 1400
    return np.vstack([interp_x(y), np.zeros_like(y), interp_z(y)]).T

def Body_acc(y, v_vel):
    return  -aero_acc(v_vel) + np.array([0, g, 0])

def calc_time():
    return float((m/-g) ** (1/2) + h / calc_V(V_w, -g * m)[0])

def calc_traect(h, t_step, v_0, m, direct):
    cords = [np.array([[0, h, 0] for i in range(direct.shape[0])])]
    vels = [np.array(direct / vec_len(direct) * v_0)]
    vall = [vels[0] + Wind_vel(h)]
    accs = [Body_acc(cords[-1][:, 1], vels[-1])]
    verbose_print(accs, 2)
    t_sum = [0]
    t = t_step
    max_y = h
    low_h = False
    while (cords[-1][:, 1] > 0).all():
        rn1 = cords[-1]
        windn1 = Wind_vel(rn1[:, 1])
        verbose_print(windn1, 2)
        vn1 = vels[-1] + accs[-1] * t
        valln1 = vn1 + windn1
        an1 = Body_acc(rn1[:, 1], valln1)
        verbose_print(an1, 2)
        dan1 = an1 - accs[-1]
        t_sum.append(t_sum[-1] + t)
        indices = rn1[:, 1] < 0
        vn1[indices] = 0
        valln1[indices] = 0
        an1[indices] = 0
        dan1[indices] = 0
        rn = rn1 + valln1 * t + an1 * t**2 / 2 + dan1 * t**2 / 6
        cords.append(rn)
        vels.append(vn1)
        accs.append(an1)
        vall.append(valln1)
        if cords[-1][:, 1].max() < max_y - 100:
            max_y -= 100
            verbose_print("Y: {:0.4f}".format(cords[-1][:, 1].max()), 2)
        if cords[-1][:, 1].min() < 10 and not low_h:
            t /= 10
            low_h = True
    return t_sum, np.array(cords), np.array(vall), np.array(accs)



t_step = calc_time() / 2000
if v_0 > calc_V(V_w, -g * m)[0] * 5:
    v_0 = calc_V(V_w, -g * m)[0] * 5
    verbose_print("Начальная скорость слишком велика, "+
            "скорость изменена на {:0.3f}".format(v_0), 2)
direct = np.array([[1, 0, 1]]).reshape(-1, 3)
t, cords, vels, accs = calc_traect(h, t_step, v_0, m, direct)

res = cords[-1, 0]

verbose_print("Результат: ({:0.3f}, {:0.3f}), Угол {:0.3f}".format(-res[0],
                                            -res[2], calc_angle(direct)[0]), 1)

res = pd.DataFrame(np.hstack([np.array(t).reshape(-1, 1),
                            cords[:, 0] - res,
                            vels[:, 0]]),
                            columns=('T', 'X', 'Y', 'Z', 'Vx', 'Vy', 'Vz'))
res.to_csv(args.output, index=None)

if args.picture is not None:
    from matplotlib import pyplot as plt
    plt.rc('figure', figsize=(15, 12))
    k = 0
    for i in [cords - res, vels, accs]:
        for j in range(3):
            plt.subplot(3, 3, k * 3 + j + 1)
            plt.plot(t, i[:, 0, j], label='speed' if k == 1 else None)
            if k == 1 and j != 1:
                x = cords[:, 0, 1]
                x[x < 0] = 0
                plt.plot(t, [interp_x, None, interp_z][j](x), label='wind')
                plt.legend()
            plt.title(['Координата', 'Скорость', 'Ускорение'][k] + ' по '
                                                        + ['X', 'Y', 'Z'][j])
        k += 1
    plt.tight_layout()
    plt.savefig(args.picture)

