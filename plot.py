import seaborn as sns
import matplotlib.pyplot as plt
import pickle

if __name__ == '__main__':
    with open("./dataset/Harmonic_1500.pkl", "rb") as fh:
        hd = pickle.load(fh)
    with open("./dataset/TV_1500.pkl", "rb") as fh:
        td = pickle.load(fh)
    with open("./dataset/Exponential_1500.pkl", "rb") as fh:
        ed = pickle.load(fh)
    # name = "TV"
    # with open("./dataset/loss_%s_dt0.1_iter1500.pkl" %name, "rb") as fh:
    #     dt1 = pickle.load(fh)
    # with open("./dataset/loss_%s_dt0.05_iter1500.pkl" %name, "rb") as fh:
    #     dt2 = pickle.load(fh)
    # with open("./dataset/loss_%s_dt0.01_iter1500.pkl" %name, "rb") as fh:
    #     dt3 = pickle.load(fh)
    # with open("./dataset/loss_%s_dt0.001_iter1500.pkl" %name, "rb") as fh:
    #     dt4 = pickle.load(fh)

    himg, hnoisy, hpsnr, hssim, hnrmse, hloss = hd['img'], hd['noisy'], hd['psnr'], hd['ssim'], hd['nrmse'], hd['loss']
    timg, tnoisy, tpsnr, tssim, tnrmse, tloss = td['img'], td['noisy'], td['psnr'], td['ssim'], td['nrmse'], td['loss']
    eimg, enoisy, epsnr, essim, enrmse, eloss = ed['img'], ed['noisy'], ed['psnr'], ed['ssim'], ed['nrmse'], ed['loss']



    plt.figure()
    plt.imshow(hnoisy)
    plt.savefig('results/noisy.png')
    plt.figure()
    plt.imshow(himg)
    plt.savefig('results/hu.png')
    plt.figure()
    plt.imshow(timg)
    plt.savefig('results/tu.png')
    plt.figure()
    plt.imshow(eimg)
    plt.savefig('results/eu.png')

    plt.figure()
    plt.grid(True)
    plt.plot(hloss, linewidth=2.0)
    plt.savefig('results/hloss.png')
    plt.figure()
    plt.grid(True)
    plt.plot(tloss, linewidth=2.0)
    plt.savefig('results/tloss.png')
    plt.figure()
    plt.grid(True)
    plt.plot(eloss, linewidth=2.0)
    plt.savefig('results/eloss.png')

    plt.figure()
    plt.grid(True)
    plt.plot(hpsnr, linewidth=2.0, label="Harmonic PSNR")
    plt.plot(tpsnr, linewidth=2.0, label="TV PSNR")
    plt.plot(epsnr, linewidth=2.0, label="Exponential PSNR")
    plt.legend(loc='lower right')
    plt.ylabel("PSNR")
    plt.xlabel("Iterations")
    plt.savefig('results/psnr.png')

    plt.figure()
    plt.grid(True)
    plt.plot(hnrmse, linewidth=2.0, label="Harmonic NRMSE")
    plt.plot(tnrmse, linewidth=2.0, label="TV NRMSE")
    plt.plot(enrmse, linewidth=2.0, label="Exponential NRMSE")
    plt.legend(loc='upper right')
    plt.ylabel("NRMSE")
    plt.xlabel("Iterations")
    plt.savefig('results/nrmse.png')

    plt.figure()
    plt.grid(True)
    plt.plot(hssim, linewidth=2.0, label="Harmonic SSIM")
    plt.plot(tssim, linewidth=2.0, label="TV SSIM")
    plt.plot(essim, linewidth=2.0, label="Exponential SSIM")
    plt.legend(loc='lower right')
    plt.ylabel("SSIM")
    plt.xlabel("Iterations")
    plt.savefig('results/ssim.png')

    # plt.figure()
    # plt.grid(True)
    # plt.axis([-100, 1550, 1200, 3500])
    # p1 = plt.plot(dt1['loss'], label="dt=%s"%(dt1['dt']), linewidth=2.0)
    # p2 = plt.plot(dt2['loss'], label="dt=%s"%(dt2['dt']), linewidth=2.0)
    # p3 = plt.plot(dt3['loss'], label="dt=%s"%(dt3['dt']), linewidth=2.0)
    # p4 = plt.plot(dt4['loss'], label="dt=%s"%(dt4['dt']), linewidth=2.0)
    # plt.legend(loc='upper right')
    # plt.savefig('results/loss_plot%s.png' %name)
