import numpy
import quantities as pq
import matplotlib.pyplot as plt
import copy
from scipy.optimize import curve_fit
import neo
import elephant.unitary_event_analysis as ue
import elephant.spike_train_generation as stg

def _plot_UE(data,Js_dict,sig_level,binsize,winsize,winstep, pattern_hash,N,args,add_epochs = []):
    """
    Examples:
    ---------
    dict_args = {'events':{'SO':[100*pq.ms]},
     'save_fig': True,
     'path_filename_format':'UE1.pdf',
     'showfig':True,
     'suptitle':True,
     'figsize':(12,10),
    'unit_ids':[10, 19, 20],
    'ch_ids':[1,3,4],
    'fontsize':15,
    'linewidth':2,
    'set_xticks' :False'}
    'marker_size':8,
    """
    import matplotlib.pylab as plt
    t_start = data[0][0].t_start
    t_stop = data[0][0].t_stop

    t_winpos = ue._winpos(t_start,t_stop,winsize,winstep)
    Js_sig = ue.jointJ(sig_level)
    num_tr = len(data)
    pat = ue.inverse_hash_from_pattern(pattern_hash, N)
    events = args['events']

    # figure format
    figsize = args['figsize']
    if 'top' in args.keys():
        top = args['top']
    else:
        top=.90
    if 'bottom' in args.keys():
        bottom = args['bottom']
    else:
        bottom=.05
    if 'right' in args.keys():
        right = args['right']
    else:
        right=.95
    if 'left' in args.keys():
        left = args['left']
    else:
        left=.1

    if 'hspace' in args.keys():
        hspace = args['hspace']
    else:
        hspace=.5
    if 'wspace' in args.keys():
        wspace = args['wspace']
    else:
        wspace=.5

    if 'fontsize' in args.keys():
        fsize = args['fontsize']
    else:
        fsize = 20
    if 'unit_ids' in args.keys():
        unit_real_ids = args['unit_ids']
        if len(unit_real_ids) != N:
            raise ValueError('length of unit_ids should be equal to number of neurons!')
    else:
        unit_real_ids = numpy.arange(1,N+1,1)
    if 'ch_ids' in args.keys():
        ch_real_ids = args['ch_ids']
        if len(ch_real_ids) != N:
            raise ValueError('length of ch_ids should be equal to number of neurons!')
    else:
        ch_real_ids = []

    if 'showfig' in args.keys():
        showfig = args['showfig']
    else:
        showfig = False
    if 'linewidth' in args.keys():
        lw = args['linewidth']
    else:
        lw = 2

    if 'S_ylim' in args.keys():
        S_ylim = args['S_ylim']
    else:
        S_ylim = [-3,3]

    if 'marker_size' in args.keys():
        ms = args['marker_size']
    else:
        ms = 8

    if add_epochs != []:
        coincrate = add_epochs['coincrate']
        backgroundrate = add_epochs['backgroundrate']
        num_row = 6
    else:
        num_row = 5
    num_col = 1
    ls = '-'
    alpha = 0.5
    plt.figure(1,figsize = figsize)
    if args['suptitle'] == True:
        plt.suptitle("Spike Pattern:"+ str((pat.T)[0]),fontsize = 20)
    print 'plotting UEs ...'
    plt.subplots_adjust(top=top, right=right, left=left, bottom=bottom, hspace=hspace , wspace=wspace)
    ax = plt.subplot(num_row,1,1)
    ax.set_title('Unitary Events',fontsize = 20,color = 'r')
    for n in range(N):
        for tr,data_tr in enumerate(data):
            plt.plot(data_tr[n].rescale('ms').magnitude, numpy.ones_like(data_tr[n].magnitude)*tr + n*(num_tr + 1) + 1, '.', markersize=0.5,color = 'k')
            sig_idx_win = numpy.where(Js_dict['Js']>= Js_sig)[0]
            if len(sig_idx_win)>0:
                x = numpy.unique(Js_dict['indices']['trial'+str(tr)])
                if len(x)>0:
                    xx = []
                    for j in sig_idx_win:
                        xx =numpy.append(xx,x[numpy.where((x*binsize>=t_winpos[j]) &(x*binsize<t_winpos[j] + winsize))])
                    plt.plot(
                        numpy.unique(xx)*binsize, numpy.ones_like(numpy.unique(xx))*tr + n*(num_tr + 1) + 1,
                        ms=ms, marker = 's', ls = '',mfc='none', mec='r')
        plt.axhline((tr + 2)*(n+1) ,lw = 2, color = 'k')
    y_ticks_pos = numpy.arange(num_tr/2 + 1,N*(num_tr+1), num_tr+1)
    plt.yticks(y_ticks_pos)
    plt.gca().set_yticklabels(unit_real_ids,fontsize = fsize)
    for ch_cnt, ch_id in enumerate(ch_real_ids):
        print ch_id
        plt.gca().text((max(t_winpos) + winsize).rescale('ms').magnitude,
                       y_ticks_pos[ch_cnt],'CH-'+str(ch_id),fontsize = fsize)

    plt.ylim(0, (tr + 2)*(n+1) + 1)
    plt.xlim(0, (max(t_winpos) + winsize).rescale('ms').magnitude)
    plt.xticks([])
    plt.ylabel('Unit ID',fontsize = fsize)
    for key in events.keys():
        for e_val in events[key]:
            plt.axvline(e_val,ls = ls,color = 'r',lw = 2,alpha = alpha)
    if 'set_xticks' in args.keys() and args['set_xticks'] == False:
        plt.xticks([])
    print 'plotting Raw Coincidences ...'
    ax1 = plt.subplot(num_row,1,2,sharex = ax)
    ax1.set_title('Raw Coincidences',fontsize = 20,color = 'c')
    for n in range(N):
        for tr,data_tr in enumerate(data):
            plt.plot(data_tr[n].rescale('ms').magnitude,
                     numpy.ones_like(data_tr[n].magnitude)*tr + n*(num_tr + 1) + 1,
                     '.', markersize=0.5, color = 'k')
            plt.plot(
                numpy.unique(Js_dict['indices']['trial'+str(tr)])*binsize,
                numpy.ones_like(numpy.unique(Js_dict['indices']['trial'+str(tr)]))*tr + n*(num_tr + 1) + 1,
                ls = '',ms=ms, marker = 's', markerfacecolor='none', markeredgecolor='c')
        plt.axhline((tr + 2)*(n+1) ,lw = 2, color = 'k')
    plt.ylim(0, (tr + 2)*(n+1) + 1)
    plt.yticks(numpy.arange(num_tr/2 + 1,N*(num_tr+1), num_tr+1))
    plt.gca().set_yticklabels(unit_real_ids,fontsize = fsize)
    plt.xlim(0, (max(t_winpos) + winsize).rescale('ms').magnitude)
    plt.xticks([])
    plt.ylabel('Unit ID',fontsize = fsize)
    for key in events.keys():
        for e_val in events[key]:
            plt.axvline(e_val,ls = ls,color = 'r',lw = 2,alpha = alpha)

    print 'plotting PSTH ...'
    plt.subplot(num_row,1,3,sharex=ax)
    #max_val_psth = 0.*pq.Hz
    for n in range(N):
        #data_psth = []
        #for tr,data_tr in enumerate(data):
        #    data_psth.append(data_tr[p])
        #psth = ss.peth(data_psth, w = psth_width)
        #plt.plot(psth.times,psth.base/float(num_tr)/psth_width.rescale('s'), label = 'unit '+str(unit_real_ids[p]))
        #max_val_psth = max(max_val_psth, max((psth.base/float(num_tr)/psth_width.rescale('s')).magnitude))
        plt.plot(t_winpos + winsize/2.,Js_dict['rate_avg'][:,n].rescale('Hz'),label = 'unit '+str(unit_real_ids[n]),lw = lw)
        #max_val_psth = max(max_val_psth, max(Js_dict['rate_avg'][:,n].rescale('Hz')))
    plt.ylabel('Rate [Hz]',fontsize = fsize)
    plt.xlim(0, (max(t_winpos) + winsize).rescale('ms').magnitude)
    max_val_psth = plt.gca().get_ylim()[1]
    plt.ylim(0, max_val_psth)
    plt.yticks([0, int(max_val_psth/2),int(max_val_psth)],fontsize = fsize)
    plt.legend(bbox_to_anchor=(1.12, 1.05), fancybox=True, shadow=True)
    for key in events.keys():
        for e_val in events[key]:
            plt.axvline(e_val,ls = ls,color = 'r',lw = lw,alpha = alpha)

    if 'set_xticks' in args.keys() and args['set_xticks'] == False:
        plt.xticks([])
    print 'plotting emp. and exp. coincidences rate ...'
    plt.subplot(num_row,1,4,sharex=ax)
    plt.plot(t_winpos + winsize/2.,Js_dict['n_emp'],label = 'empirical',lw = lw,color = 'c')
    plt.plot(t_winpos + winsize/2.,Js_dict['n_exp'],label = 'expected',lw = lw,color = 'm')
    plt.xlim(0, (max(t_winpos) + winsize).rescale('ms').magnitude)
    plt.ylabel('# Coinc.',fontsize = fsize)
    plt.legend(bbox_to_anchor=(1.12, 1.05), fancybox=True, shadow=True)
    YTicks = plt.ylim(0,int(max(max(Js_dict['n_emp']), max(Js_dict['n_exp']))))
    plt.yticks([0,YTicks[1]],fontsize = fsize)
    for key in events.keys():
        for e_val in events[key]:
            plt.axvline(e_val,ls = ls,color = 'r',lw = 2,alpha = alpha)
    if 'set_xticks' in args.keys() and args['set_xticks'] == False:
        plt.xticks([])

    print 'plotting Surprise ...'
    plt.subplot(num_row,1,5,sharex=ax)
    plt.plot(t_winpos + winsize/2., Js_dict['Js'],lw = lw,color = 'k')
    plt.xlim(0, (max(t_winpos) + winsize).rescale('ms').magnitude)
    plt.axhline(Js_sig,ls = '-', color = 'gray')
    plt.axhline(-Js_sig,ls = '-', color = 'gray')
    plt.gca().text(10,Js_sig + 0.2, str(int(sig_level*100))+'%',fontsize = fsize-2,color = 'gray')
    plt.xticks(t_winpos.magnitude[::len(t_winpos)/10])
    plt.yticks([-2,0,2],fontsize = fsize)
    plt.ylabel('S',fontsize = fsize)
    plt.xlabel('Time [ms]', fontsize = fsize)
    plt.ylim(S_ylim)
    for key in events.keys():
        for e_val in events[key]:
            plt.axvline(e_val,ls = ls,color = 'r',lw = lw,alpha = alpha)
            plt.gca().text(e_val - 10*pq.ms,2*S_ylim[0],key,fontsize = fsize,color = 'r')
    if 'set_xticks' in args.keys() and args['set_xticks'] == False:
        plt.xticks([])

    if add_epochs != []:
        plt.subplot(num_row,1,6,sharex=ax)
        plt.plot(coincrate,lw = lw,color = 'c')
        plt.plot(backgroundrate,lw = lw,color = 'm')
        plt.xlim(0, (max(t_winpos) + winsize).rescale('ms').magnitude)
        plt.ylim(plt.gca().get_ylim()[0]-2,plt.gca().get_ylim()[1]+2)
    if args['save_fig'] == True:
        plt.savefig(args['path_filename_format'])
        if showfig == False:
            plt.cla()
            plt.close()
   # plt.xticks(t_winpos.magnitude)

    if showfig == True:
        plt.show()
