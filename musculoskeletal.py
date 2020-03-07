import collections
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve
from sklearn.linear_model import Ridge
from scipy.special import expit
from scipy.integrate import solve_ivp

class HillTypeMuscle:
    """
    Damped Hill-type muscle model adapted from Millard et al. (2013). The
    dynamic model is defined in terms of normalized length and velocity.
    To model a particular muscle, scale factors are needed for force, CE
    length, and SE length. These are given as constructor arguments.
    """

    def __init__(self, f0M, resting_length_muscle, resting_length_tendon):
        """
        :param f0M: maximum isometric force
        :param resting_length_muscle: actual length (m) of muscle (CE) that corresponds to
            normalized length of 1
        :param resting_length_tendon: actual length of tendon (m) that corresponds to
            normalized length of 1
        """
        self.f0M = f0M
        self.resting_length_muscle = resting_length_muscle
        self.resting_length_tendon = resting_length_tendon
        print ("making Hill-type stuff in musculo-file")

    def norm_tendon_length(self, muscle_tendon_length, normalized_muscle_length):
        """
        :param muscle_tendon_length: non-normalized length of the full muscle-tendon (0.4)
            complex (typically found from joint angles and musculoskeletal geometry)
        :param normalized_muscle_length: normalized length of the contractile element
            (the state variable of the muscle model)
        :return: normalized length of the tendon
        """
        return (muscle_tendon_length - self.resting_length_muscle * normalized_muscle_length) / self.resting_length_tendon

    def get_force(self, total_length, norm_muscle_length):
        """
        :param total_length: muscle-tendon length (m)
        :param norm_muscle_length: normalized length of muscle (the state variable)
        :return: muscle tension (N)
        """
        return self.f0M * force_length_tendon(self.norm_tendon_length(total_length, norm_muscle_length))


def get_velocity(a, lm, lt):
    """
    :param a: activation (between 0 and 1)
    :param lm: normalized length of muscle (contractile element)
    :param lt: normalized length of tendon (series elastic element)
    :return: normalized lengthening velocity of muscle (contractile element)
    """
    beta = 0.1 # damping coefficient (see damped model in Millard et al.)
    def f(vm):
        p1 = a*force_length_muscle(lm)*force_velocity_muscle(vm)
        # print(p1)
        p2 = force_length_parallel(lm) + beta
        # print(p2)
        p3 = force_length_tendon(lt)
        # print (p3)
        return ((p1 + p2) - p3)
    return fsolve(f, 0)

    # WRITE CODE HERE TO CALCULATE VELOCITY


def force_length_tendon(lt):
    """
    :param lt: normalized length of tendon (series elastic element)
    :return: normalized tension produced by tendon
    """
    # WRITE CODE HERE
    if isinstance(lt, float) or isinstance(lt, int):
        if lt < 1:
            return 0
        elif lt >= 1:
            return ((3*(lt - 1)**2)/(-0.4+lt))
    

    lenTend = 0*lt
    for i in range(len(lt)):
        if lt[i] < 1:
            lenTend[i] = 0
        if lt[i] >= 1:
            lenTend[i] = (10*(lt[i] - 1)+240*(lt[i] - 1)**2)
    return lenTend


def force_length_parallel(lm):
    """
    :param lm: normalized length of muscle (contractile element)
    :return: normalized force produced by parallel elastic element
    """
    # WRITE CODE HERE
    if isinstance(lm, float) or isinstance(lm, int):
        if lm < 1:
            return 0
        else:
            return ((3*(lm - 1)**2)/(-0.4+lm))
    
    lenPar = lm*0
    for i in range(len(lm)):
        if lm[i] < 1:
            lenPar[i] = 0
        elif lm[i] >= 1:
            lenPar[i] = ((3*(lm[i] - 1)**2)/(-0.4+lm[i]))
    return lenPar

def plot_curves():
    """
    Plot force-length, force-velocity, SE, and PE curves.
    """
    lm = np.arange(0, 1.8, .01)
    vm = np.arange(-1.2, 1.2, .01)
    lt = np.arange(0, 1.07, .01)
    plt.subplot(2,1,1)
    plt.plot(lm, force_length_muscle(lm), 'r')
    plt.plot(lm, force_length_parallel(lm), 'g')
    plt.plot(lt, force_length_tendon(lt), 'b')
    plt.legend(('CE', 'PE', 'SE'))
    plt.xlabel('Normalized length')
    plt.ylabel('Force scale factor')
    plt.subplot(2, 1, 2)
    plt.plot(vm, force_velocity_muscle(vm), 'k')
    plt.xlabel('Normalized muscle velocity')
    plt.ylabel('Force scale factor')
    plt.tight_layout()
    plt.show()


class Gaussian:
    def __init__(self, mu, sigma):
        self.mu = mu
        self.sigma = sigma

    def __call__(self, x):
        return np.exp(-(x-self.mu)**2/2/self.sigma**2)


class Sigmoid:
    def __init__(self, mu, sigma):
        self.mu = mu
        self.sigma = sigma

    def __call__(self, x):
        return expit((x-self.mu) / self.sigma)


class Regression():
    """
    1D regression model with Gaussian basis functions.
    """

    def __init__(self, x, t, centres, width, regularization_weight=1e-6, sigmoids=False):
        """
        :param x: samples of an independent variable
        :param t: corresponding samples of a dependent variable
        :param centres: a vector of Gaussian centres (should have similar range of values as x)
        :param width: sigma parameter of Gaussians
        :param regularization_weight: regularization strength parameter
        """
        if sigmoids:
            self.basis_functions = [Sigmoid(centre, width) for centre in centres]
        else:
            self.basis_functions = [Gaussian(centre, width) for centre in centres]
        self.ridge = Ridge(alpha=regularization_weight, fit_intercept=False)
        self.ridge.fit(self._get_features(x), t)

    def eval(self, x):
        """
        :param x: a new (or multiple samples) of the independent variable
        :return: the value of the curve at x
        """
        return self.ridge.predict(self._get_features(x))

    def _get_features(self, x):
        if not isinstance(x, collections.Sized):
            x = [x]

        phi = np.zeros((len(x), len(self.basis_functions)))
        for i, basis_function in enumerate(self.basis_functions):
            phi[:,i] = basis_function(x)
        return phi


def get_muscle_force_velocity_regression():
    data = np.array([
        [-1.0028395556708567, 0.0024834319945283845],
        [-0.8858611825192801, 0.03218792009622429],
        [-0.5176245843258415, 0.15771090304473967],
        [-0.5232565269687035, 0.16930496922242444],
        [-0.29749770052593094, 0.2899790099290114],
        [-0.2828848376217543, 0.3545364496120378],
        [-0.1801231103040022, 0.3892195938775034],
        [-0.08494610976156225, 0.5927831890757294],
        [-0.10185137142991896, 0.6259097662790973],
        [-0.0326643239546236, 0.7682365981934388],
        [-0.020787245583830716, 0.8526638522676352],
        [0.0028442725407418212, 0.9999952831301149],
        [0.014617579774061973, 1.0662107025777694],
        [0.04058866536166583, 1.124136223202283],
        [0.026390887007381902, 1.132426122025424],
        [0.021070257776939272, 1.1986556920827338],
        [0.05844673474682183, 1.2582274002971627],
        [0.09900238201929201, 1.3757434966156459],
        [0.1020023112662436, 1.4022310794556732],
        [0.10055894908138963, 1.1489210160137733],
        [0.1946227683309354, 1.1571212943090965],
        [0.3313459588217258, 1.152041225442796],
        [0.5510200231126625, 1.204839508502158]
    ])

    velocity = data[:,0]
    force = data[:,1]

    centres = np.arange(-1, 0, .2)
    width = .15
    result = Regression(velocity, force, centres, width, .1, sigmoids=True)

    return result


def get_muscle_force_length_regression():
    """
    CE force-length data samples from Winters et al. (2011) Figure 3C,
    normalized so that max force is ~1 and length at max force is ~1.
    The sampples were taken form the paper with WebPlotDigitizer, and
    cut-and-pasted here.
    WRITE CODE HERE 1) Use WebPlotDigitizer to extract force-length points
    from Winters et al. (2011) Figure 3C, which is on Learn. Click
    "View Data", select all, cut, and paste below. 2) Normalize the data
    so optimal length = 1 and peak = 1. 3) Return a Regression object that
    uses Gaussian basis functions. 
    """
    
    TAActive = np.array([
        [72.41286863270777, 119.62858321303361],
        [73.41823056300268, 118.39265828005773],
        [71.40750670241287, 109.63373891524024],
        [70.48257372654156, 94.25386677665497],
        [68.9946380697051, 86.41534336976696],
        [69.43699731903484, 84.41307486079604],
        [66.98391420911528, 77.34873169725716],
        [68.23056300268095, 73.34233862652093],
        [68.51206434316353, 74.57166426067229],
        [68.79356568364611, 74.57022066405443],
        [70.0402144772118, 71.33305836254897],
        [71.28686327077747, 71.94204990719734],
        [66.8632707774799, 68.5801196122912],
        [67.78820375335121, 69.34460713549186],
        [68.83378016085791, 69.18539905135079],
        [70.0402144772118, 61.48690451639513],
        [68.83378016085791, 59.8007836667354],
        [68.39142091152814, 57.341513714167874],
        [67.50670241286863, 58.88451227057125],
        [67.38605898123325, 61.19282326252835],
        [65.6166219839142, 57.66343575995049],
        [65.01340482573727, 56.743452258197564],
        [66.26005361930295, 55.352443802845954],
        [65.77747989276139, 44.43184161682821],
        [64.7721179624665, 45.36007424211178],
        [64.7721179624665, 39.513920395957925],
        [66.42091152815013, 33.81315735203134],
        [66.54155495978551, 32.88946174468964],
        [65.6970509383378, 31.04763868838937],
        [64.97319034852546, 33.051350793978145],
        [65.57640750670241, 28.279026603423404],
        [65.85790884718499, 27.508352237574755],
        [63.92761394101876, 29.056712724273055],
        [63.68632707774798, 28.442565477418015],
        [64.49061662198392, 25.823056300268092],
        [63.6461126005362, 26.135079397813982],
        [63.68632707774799, 24.59641163126416],
        [63.36461126005362, 24.444215302124135],
        [63.16353887399464, 25.06063105795009],
        [63.40482573726541, 22.59785522788205],
        [63.6461126005362, 21.365848628583223],
        [63.68632707774799, 19.519488554341095],
        [63.32439678284182, 18.598267684058555],
        [63.243967828418235, 15.21406475561973],
        [62.31903485254691, 17.064961847803673],
        [61.35388739946381, 14.300680552691276],
        [61.35388739946381, 11.06991132192205],
        [62.11796246648793, 10.142916065168066],
        [62.80160857908847, 9.216333264590645],
        [62.27882037533512, 8.60362961435348],
        [62.03753351206434, 7.527943905960001],
        [62.19839142091153, 5.988657455145386],
        [62.520107238605895, 7.217776861208506],
        [62.92225201072386, 6.60032996494121],
        [63.32439678284182, 5.67519076098165],
        [60.58981233243968, 8.304598886368325],
        [60.99195710455764, 7.225613528562604],
        [61.27345844504021, 3.531862239637036],
        [59.82573726541555, 5.539286450814615],
        [59.82573726541555, 4.30851722004536],
        [59.343163538873995, 5.541761187873789],
        [59.343163538874, 4.9263765724891755],
        [58.860589812332435, 4.467312848009897],
        [59.46380697050939, 2.4642194266859008],
        [58.90080428954424, 2.1594143122293303],
        [58.37801608579088, 3.7005568158383255],
        [58.297587131367294, 1.8548154258610055],
        [58.498659517426276, 2.6230150546504376],
        [57.975871313672926, 0.779542173644046],
        [57.81501340482574, 2.6265209321509815],
        [57.533512064343164, 0.3202722210764932],
        [57.252010723860586, 2.4755619715405146],
        [56.84986595174262, 1.5545473293462635],
        [56.48793565683646, -0.1359043101670636],
        [56.447721179624665, -0.5972365436172424],
        [56.1260053619303, 0.3274902041657981],
        [56.1260053619303, -0.5955867189111075],
        [55.72386058981233, 1.5603217158177074],
        [55.321715817694376, 0.02392245823881467],
        [54.638069705093834, 0.18127448958549053],
        [54.155495978552274, 0.9529799958754381],
        [53.793565683646115, 0.18560527943907346],
        [53.47184986595175, -0.2742833573932728],
        [53.51206434316354, 0.8024334914415476],
        [52.78820375335121, 0.3446071354918274],
        [52.62734584450402, -0.4237987213858503],
        [52.184986595174266, -0.42153021241492183],
        [51.90348525469169, -0.11239430810476847],
        [51.62198391420911, -0.2647968653330395],
        [51.501340482573724, 0.04351412662404641],
        [50.656836461126005, -0.10600123736853106],
        [50.656836461126005, -0.41369354506082345],
        [50.25469168900804, -0.25778511033200857],
        [49.611260053619304, -0.2544854609197671],
        [48.92761394101876, -0.09713342957311966],
        [48.92761394101876, -0.7125180449577329],
        [48.16353887399464, -0.24706124974218824],
        [47.6005361930295, -0.2441740565064947],
        [47.43967828418231, -0.24334914415342723],
        [47.03753351206434, -0.24128686327077276],
        [46.63538873994638, -0.23922458238811828],
        [46.394101876675606, -0.23798721385850286],
        [45.428954423592494, -0.23303773974012643],
        [45.187667560321714, -0.23180037121053942],
        [44.38337801608579, -0.22767580944523047],
        [43.337801608579085, -0.22231387915033451],
        [42.85522788203754, -0.2198391420911321],
        [41.36729222520107, -0.5199010105176285],
        [40.36193029490617, -0.5147453083109781],
        [39.396782841823054, -0.35594968034644126],
        [38.39142091152815, -0.3507939781398193],
        [37.38605898123325, -0.34563827593316887]
    ])
    TAActive = TAActive/[72.41286863270777, 119.62858321303361]

    lengthA = TAActive[:,0]
    forceA = TAActive[:,1]
    
    centres = np.arange(min(lengthA), max(lengthA), 0.1)
    width = .155                                 #[-0.78820701] is force velocity at a=1, lm=1, lt=1.01, width = 0.155
    result = Regression(lengthA, forceA, centres, width, .1, sigmoids=False)

    return result



force_length_regression = get_muscle_force_length_regression()
force_velocity_regression = get_muscle_force_velocity_regression()


def force_length_muscle(lm):
    """
    :param lm: muscle (contracile element) length
    :return: force-length scale factor
    """
    return force_length_regression.eval(lm)


def force_velocity_muscle(vm):
    """
    :param vm: muscle (contractile element) velocity)
    :return: force-velocity scale factor
    """
    return np.maximum(0, force_velocity_regression.eval(vm))
lm = np.arange(0, 1.8, 1.8/100)
# vm = np.arange(-1.2, 1.2, 1/240)
lt = np.arange(0, 1.07, 1.07/100)

a = np.arange(0, 1, 1/100)
plot_curves()
# a = [1]
# lm = [1]
# lt = [1.01]
print (get_velocity(1,1,1.01))
# def ourMuscle(x):
    #need to plot the contractile element length. We have a function for the tendon length. Overall length is set at 1 normalized, or 0.4 unnormalized
    #need to plot the force produced by the muscle
def finding_lm(t, x):
    # time = np.arange(0, 2, 0.1)
    # result = np.arange(0, 2, 0.1)*0
    # velocity_store = []
    if t < 0.5:
        return get_velocity(0, x, 2-x)
    else:
        return get_velocity(1, x, 2-x)
    #     velocity_store.append(get_velocity(a, x, 2-x))
    #  velocity_store
    # return result
contractile_length = solve_ivp(finding_lm, [0,2], [1], max_step = 0.01)
myMuscle = HillTypeMuscle(100, .3, .1)


plt.figure()
plt.subplot(1,2,1)
plt.plot(contractile_length.t, contractile_length.y.T)
plt.xlabel('Time (s)')
plt.ylabel('Normalized CE length')
plt.subplot(1,2,2)
plt.plot(contractile_length.t, myMuscle.get_force(0.4, contractile_length.y.T))
plt.xlabel('Time (s)')
plt.ylabel('Normalized Tension')
plt.tight_layout()
plt.show()


print ("in musculo-file")
