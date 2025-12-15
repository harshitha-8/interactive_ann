# widgets and dashboard
l = widgets.Text(value='Interactive Simple Artificial Neural Network', layout=Layout(width='950px', height='30px'))

nepoch = widgets.IntSlider(min=0, max=100, value=0, step=1, description='$n_{epoch}$', orientation='horizontal', style={'description_width': 'initial'},
                           continuous_update=False, layout=Layout(width='300px', height='30px'))

lr = widgets.FloatLogSlider(min=-2, max=0, value=0.1, step=1.0, description='$\\eta$', orientation='horizontal', style={'description_width': 'initial'},
                           continuous_update=False, layout=Layout(width='300px', height='30px'), readout_format='.2f')

seed = widgets.IntSlider(min=0, max=100, value=1, step=1, description='$S$', orientation='horizontal', style={'description_width': 'initial'},
                           continuous_update=False, layout=Layout(width='300px', height='30px'))

weights = widgets.Checkbox(value=True, description='Weights', disabled=False, layout=Layout(width='300px', height='30px'))

forward = widgets.Checkbox(value=True, description='Forward Pass', disabled=False, layout=Layout(width='300px', height='30px'))

back = widgets.Checkbox(value=True, description='Back Propogation', disabled=False, layout=Layout(width='300px', height='30px'))

ui1 = widgets.HBox([nepoch, lr, seed],)
ui2 = widgets.HBox([weights, forward, back],)
ui = widgets.VBox([l, ui1, ui2],)

def run_plot(nepoch, lr, weights, forward, back, seed):                       # make data, fit models and plot
    min_lw = 0.5; node_r = 0.2; min_node_r = 0.20
    lw = 2.0 - min_lw; min_alpha = 0.1
    alpha = 1.0 - min_alpha
    iepoch = nepoch
    np.random.seed(seed=seed)
    x1 = 0.5; x2 = 0.2; x3 = 0.7; y = 0.3 # training data
    
    np.random.seed(seed=seed)
    
    nepoch = 1000
    
    y4 = np.zeros(nepoch); y5 = np.zeros(nepoch); y6 = np.zeros(nepoch)
    
    w14 = np.zeros(nepoch); w24 = np.zeros(nepoch); w34 = np.zeros(nepoch)
    w15 = np.zeros(nepoch); w25 = np.zeros(nepoch); w35 = np.zeros(nepoch)
    w46 = np.zeros(nepoch); w56 = np.zeros(nepoch)
    
    dw14 = np.zeros(nepoch); dw24 = np.zeros(nepoch); dw34 = np.zeros(nepoch)
    dw15 = np.zeros(nepoch); dw25 = np.zeros(nepoch); dw35 = np.zeros(nepoch)
    dw46 = np.zeros(nepoch); dw56 = np.zeros(nepoch)
    
    db4 = np.zeros(nepoch); db5 = np.zeros(nepoch); db6 = np.zeros(nepoch)
    
    b4 = np.zeros(nepoch); b5 = np.zeros(nepoch); b6 = np.zeros(nepoch)
    y4 = np.zeros(nepoch); y5 = np.zeros(nepoch); y6 = np.zeros(nepoch)
    y4in = np.zeros(nepoch); y5in = np.zeros(nepoch); y6in = np.zeros(nepoch)
    d4 = np.zeros(nepoch); d5 = np.zeros(nepoch); d6 = np.zeros(nepoch)
    d1 = np.zeros(nepoch); d2 = np.zeros(nepoch); d3 = np.zeros(nepoch)
    
    # initialize the weights - Xavier Weight Initialization 
    lower, upper = -(1.0 / np.sqrt(3.0)), (1.0 / np.sqrt(3.0)) # lower and upper bound for the weights, uses inputs to node
    #lower, upper = -(sqrt(6.0) / sqrt(3.0 + 2.0)), (sqrt(6.0) / sqrt(3.0 + 2.0)) # Normalized Xavier weights, integrates ouputs also
    w14[0] = lower + np.random.random() * (upper - lower); 
    w24[0] = lower + np.random.random() * (upper - lower); 
    w34[0] = lower + np.random.random() * (upper - lower);
    w15[0] = lower + np.random.random() * (upper - lower); 
    w25[0] = lower + np.random.random() * (upper - lower); 
    w35[0] = lower + np.random.random() * (upper - lower);
    
    lower, upper = -(1.0 / np.sqrt(2.0)), (1.0 / np.sqrt(2.0))
    #lower, upper = -(sqrt(6.0) / sqrt(2.0 + 1.0)), (sqrt(6.0) / sqrt(2.0 + 1.0)) # Normalized Xavier weights, integrates ouputs also

    w46[0] = lower + np.random.random() * (upper - lower); 
    w56[0] = lower + np.random.random() * (upper - lower);     

    #b4[0] = np.random.random(); b5[0] = np.random.random(); b6[0] = np.random.random()
    b4[0] = (np.random.random()-0.5)*0.5; b5[0] = (np.random.random()-0.5)*0.5; b6[0] = (np.random.random()-0.5)*0.5; # small random value    

    for i in range(0,nepoch):
    
    # forward pass of model
        y4in[i] = w14[i]*x1 + w24[i]*x2 + w34[i]*x3 + b4[i]; 
        y4[i] = 1.0/(1 + math.exp(-1*y4in[i]))
        
        y5in[i] = w15[i]*x1 + w25[i]*x2 + w35[i]*x3 + b5[i]
        y5[i] = 1.0/(1 + math.exp(-1*y5in[i]))
        
        y6in[i] = w46[i]*y4[i] + w56[i]*y5[i] + b6[i]
        y6[i] = y6in[i]
    #    y6[i] = 1.0/(1 + math.exp(-1*y6in[i])) # sgimoid / logistic activation at o6 
    
    # back propagate the error through the nodes
    #    d6[i] = y6[i]*(1-y6[i])*(y-y6[i]) # sgimoid / logistic activation at o6 
        d6[i] = (y-y6[i]) # identity activation o at o6
        d5[i] = y5[i]*(1-y5[i])*w56[i]*d6[i]; d4[i] = y4[i]*(1-y4[i])*w46[i]*d6[i]
        d1[i] = w14[i]*d4[i] + w15[i]*d5[i]; d2[i] = w24[i]*d4[i] + w25[i]*d5[i]; d3[i] = w34[i]*d4[i] + w35[i]*d5[i] # identity and 2 paths
    
    # calculate the change in weights
        if i < nepoch - 1:
            dw14[i] = lr*d4[i]*x1; dw24[i] = lr*d4[i]*x2; dw34[i] = lr*d4[i]*x3 
            dw15[i] = lr*d5[i]*x1; dw25[i] = lr*d5[i]*x2; dw35[i] = lr*d5[i]*x3
            dw46[i] = lr*d6[i]*y4[i]; dw56[i] = lr*d6[i]*y5[i] 
            
            db4[i] = lr*d4[i]; db5[i] = lr*d5[i]; db6[i] = lr*d6[i];
    
            w14[i+1] = w14[i] + dw14[i]; w24[i+1] = w24[i] + dw24[i]; w34[i+1] = w34[i] + dw34[i] 
            w15[i+1] = w15[i] + dw15[i]; w25[i+1] = w25[i] + dw25[i]; w35[i+1] = w35[i] + dw35[i] 
            w46[i+1] = w46[i] + dw46[i]; w56[i+1] = w56[i] + dw56[i]
    
            b4[i+1] = b4[i] + db4[i]; b5[i+1] = b5[i] + db5[i]; b6[i+1] = b6[i] + db6[i] 
    
    dx = -0.21; dy = -0.09; edge = 1.0
    
    o6x = 17; o6y =5; h5x = 10; h5y = 3.5; h4x = 10; h4y = 6.5
    i1x = 3; i1y = 9.0; i2x = 3; i2y = 5; i3x = 3; i3y = 1.0; buffer = 0.5
    
    max_x = np.max(np.abs([x1,x2,x3]))
    max_y = np.max(np.abs([y4[iepoch],y5[iepoch],y6[iepoch]]))
    max_dO = np.max(np.abs([d1[iepoch],d2[iepoch],d3[iepoch]]))
    max_d = np.max(np.abs([d1[iepoch],d2[iepoch],d3[iepoch],d4[iepoch],d5[iepoch],d6[iepoch]]))
    max_d_trans = 1.0/abs(math.log(max_d)+0.00001)
    max_signal = np.max(np.abs([x1*w14[iepoch],x2*w24[iepoch],x3*w34[iepoch],
                                x1*w15[iepoch],x2*w25[iepoch],x3*w35[iepoch],
                                y4[iepoch],y5[iepoch],y6[iepoch]]))
    
    plt.subplot(111)
    plt.gca().set_axis_off()
    
    if (weights == True and forward == False and back == False) or (weights == False and forward == True and back == False):
        
        circle_i1 = plt.Circle((i1x,i1y), node_r*abs(x1)/max_x+min_node_r, fill=False, edgecolor = 'black',lw=2,zorder=100); plt.annotate(r' $I_1$',(i1x+dx,i1y+dy),zorder=110) 
        circle_i1b = plt.Circle((i1x,i1y), node_r*1.5*abs(x1)/max_x+min_node_r, fill=True, facecolor = 'white',edgecolor = None,lw=1,zorder=10)
        plt.gca().add_patch(circle_i1); plt.gca().add_patch(circle_i1b)
    
        circle_i2 = plt.Circle((i2x,i2y), node_r*abs(x2)/max_x+min_node_r, fill=False, edgecolor = 'black',lw=2,zorder=100); plt.annotate(r' $I_2$',(i2x+dx,i2y+dy),zorder=110) 
        circle_i2b = plt.Circle((i2x,i2y), node_r*1.5*abs(x2)/max_x+min_node_r, fill=True, facecolor = 'white',edgecolor = None,lw=1,zorder=10)
        plt.gca().add_patch(circle_i2); plt.gca().add_patch(circle_i2b)
    
        circle_i3 = plt.Circle((i3x,i3y), node_r*abs(x3)/max_x+min_node_r, fill=False, edgecolor = 'black',lw=2,zorder=100); plt.annotate(r' $I_3$',(i3x+dx,i3y+dy),zorder=110) 
        circle_i3b = plt.Circle((i3x,i3y), node_r*1.5*abs(x3)/max_x+min_node_r, fill=True, facecolor = 'white',edgecolor = None,lw=1,zorder=10)
        plt.gca().add_patch(circle_i3); plt.gca().add_patch(circle_i3b)
        
        circle_h4 = plt.Circle((h4x,h4y), node_r*abs(y4[iepoch])/max_y+min_node_r,fill=True,facecolor = 'red',alpha=0.5,edgecolor = 'black',lw=2,zorder=100); plt.annotate(r'$H_4$',(h4x+dx,h4y+dy),zorder=110)
        circle_h4o = plt.Circle((h4x,h4y), node_r*abs(y4[iepoch])/max_y+min_node_r,fill=False,alpha=1.0,edgecolor = 'black',lw=2,zorder=105) 
        circle_h4b = plt.Circle((h4x,h4y), node_r*1.5*abs(y4[iepoch])/max_y+min_node_r, fill=True, facecolor = 'white',edgecolor = None,lw=1,zorder=10)
        plt.gca().add_patch(circle_h4); plt.gca().add_patch(circle_h4b); plt.gca().add_patch(circle_h4o)
    
        circle_h5 = plt.Circle((h5x,h5y), node_r*abs(y5[iepoch])/max_y+min_node_r, fill=True,facecolor = 'blue',alpha=0.5,edgecolor = 'black',lw=2,zorder=100); plt.annotate(r'$H_5$',(h5x+dx,h5y+dy),zorder=110)
        circle_h5o = plt.Circle((h5x,h5y), node_r*abs(y5[iepoch])/max_y+min_node_r, fill=False,alpha=1.0,edgecolor = 'black',lw=2,zorder=105) 
        circle_h5b = plt.Circle((h5x,h5y), node_r*1.5*abs(y5[iepoch])/max_y+min_node_r, fill=True, facecolor = 'white',edgecolor = None,lw=1,zorder=10);
        plt.gca().add_patch(circle_h5); plt.gca().add_patch(circle_h5b); plt.gca().add_patch(circle_h5o)
    
        circle_o6 = plt.Circle((o6x,o6y), node_r*abs(y6[iepoch])/max_y+min_node_r,fill=True,facecolor = 'orange',alpha=0.5,edgecolor = 'black',lw=2,zorder=100); plt.annotate(r'$O_6$',(o6x+dx,o6y+dy),zorder=110)
        circle_o6o = plt.Circle((o6x,o6y), node_r*abs(y6[iepoch])/max_y+min_node_r,fill=False,alpha=1.0,edgecolor = 'black',lw=2,zorder=105)
        circle_o6b = plt.Circle((o6x,o6y), node_r*1.5*abs(y6[iepoch])/max_y+min_node_r, fill=True, facecolor = 'white',edgecolor = None,lw=1,zorder=10);
        plt.gca().add_patch(circle_o6); plt.gca().add_patch(circle_o6b); plt.gca().add_patch(circle_o6o)
  
    if weights == False and forward == False and back == True:
        i1_r = (1.0/abs(math.log(abs(d1[iepoch]))+0.00001)/max_d_trans)*node_r+min_node_r
        circle_i1 = plt.Circle((i1x,i1y), i1_r, fill=False, edgecolor = 'black',lw=2,zorder=100); plt.annotate(r' $I_1$',(i1x+dx,i1y+dy),zorder=110) 
        circle_i1b = plt.Circle((i1x,i1y), i1_r*1.5, fill=True, facecolor = 'white',edgecolor = None,lw=1,zorder=10)
        plt.gca().add_patch(circle_i1); plt.gca().add_patch(circle_i1b)

        i2_r = (1.0/abs(math.log(abs(d2[iepoch]))+0.00001)/max_d_trans)*node_r+min_node_r
        circle_i2 = plt.Circle((i2x,i2y), i2_r, fill=False, edgecolor = 'black',lw=2,zorder=100); plt.annotate(r' $I_2$',(i2x+dx,i2y+dy),zorder=110) 
        circle_i2b = plt.Circle((i2x,i2y), i2_r*1.5, fill=True, facecolor = 'white',edgecolor = None,lw=1,zorder=10)
        plt.gca().add_patch(circle_i2); plt.gca().add_patch(circle_i2b)
    
        i3_r = (1.0/abs(math.log(abs(d3[iepoch]))+0.00001)/max_d_trans)*node_r+min_node_r
        circle_i3 = plt.Circle((i3x,i3y), i3_r, fill=False, edgecolor = 'black',lw=2,zorder=100); plt.annotate(r' $I_3$',(i3x+dx,i3y+dy),zorder=110) 
        circle_i3b = plt.Circle((i3x,i3y), i3_r*1.5*abs(x3)/max_x+min_node_r, fill=True, facecolor = 'white',edgecolor = None,lw=1,zorder=10)
        plt.gca().add_patch(circle_i3); plt.gca().add_patch(circle_i3b)
        
        h4_r = (1.0/abs(math.log(abs(d4[iepoch]))+0.00001)/max_d_trans)*node_r+min_node_r
        circle_h4 = plt.Circle((h4x,h4y), h4_r,fill=True,facecolor = 'red',alpha=0.5,edgecolor = 'black',lw=2,zorder=100); plt.annotate(r'$H_4$',(h4x+dx,h4y+dy),zorder=110)
        circle_h4o = plt.Circle((h4x,h4y), h4_r,fill=False,alpha=1.0,edgecolor = 'black',lw=2,zorder=105) 
        circle_h4b = plt.Circle((h4x,h4y), h4_r*1.5, fill=True, facecolor = 'white',edgecolor = None,lw=1,zorder=10)
        plt.gca().add_patch(circle_h4); plt.gca().add_patch(circle_h4b); plt.gca().add_patch(circle_h4o)
    
        h5_r = (1.0/abs(math.log(abs(d5[iepoch]))+0.00001)/max_d_trans)*node_r+min_node_r
        circle_h5 = plt.Circle((h5x,h5y), h5_r, fill=True,facecolor = 'blue',alpha=0.5,edgecolor = 'black',lw=2,zorder=100); plt.annotate(r'$H_5$',(h5x+dx,h5y+dy),zorder=110)
        circle_h5o = plt.Circle((h5x,h5y), h5_r, fill=False,alpha=1.0,edgecolor = 'black',lw=2,zorder=105) 
        circle_h5b = plt.Circle((h5x,h5y), h5_r*1.5, fill=True, facecolor = 'white',edgecolor = None,lw=1,zorder=10);
        plt.gca().add_patch(circle_h5); plt.gca().add_patch(circle_h5b); plt.gca().add_patch(circle_h5o)
    
        h6_r = (1.0/abs(math.log(abs(d6[iepoch]))+0.00001)/max_d_trans)*node_r+min_node_r
        circle_o6 = plt.Circle((o6x,o6y), h6_r,fill=True,facecolor = 'orange',alpha=0.5,edgecolor = 'black',lw=2,zorder=100); plt.annotate(r'$O_6$',(o6x+dx,o6y+dy),zorder=110)
        circle_o6o = plt.Circle((o6x,o6y), h6_r,fill=False,alpha=1.0,edgecolor = 'black',lw=2,zorder=105)
        circle_o6b = plt.Circle((o6x,o6y), h6_r*1.5, fill=True, facecolor = 'white',edgecolor = None,lw=1,zorder=10);
        plt.gca().add_patch(circle_o6); plt.gca().add_patch(circle_o6b); plt.gca().add_patch(circle_o6o)
   
    if (weights == True and forward == True and back == True) or (weights == False and forward == False and back == False):

        circle_i1 = plt.Circle((i1x,i1y), node_r, fill=False, edgecolor = 'black',lw=2,zorder=100); plt.annotate(r' $I_1$',(i1x+dx,i1y+dy),zorder=110) 
        circle_i1b = plt.Circle((i1x,i1y), node_r*1.5, fill=True, facecolor = 'white',edgecolor = None,lw=1,zorder=10)
        plt.gca().add_patch(circle_i1); plt.gca().add_patch(circle_i1b)
    
        circle_i2 = plt.Circle((i2x,i2y), node_r, fill=False, edgecolor = 'black',lw=2,zorder=100); plt.annotate(r' $I_2$',(i2x+dx,i2y+dy),zorder=110) 
        circle_i2b = plt.Circle((i2x,i2y), node_r*1.5, fill=True, facecolor = 'white',edgecolor = None,lw=1,zorder=10)
        plt.gca().add_patch(circle_i2); plt.gca().add_patch(circle_i2b)
    
        circle_i3 = plt.Circle((i3x,i3y), node_r, fill=False, edgecolor = 'black',lw=2,zorder=100); plt.annotate(r' $I_3$',(i3x+dx,i3y+dy),zorder=110) 
        circle_i3b = plt.Circle((i3x,i3y), node_r*1.5, fill=True, facecolor = 'white',edgecolor = None,lw=1,zorder=10)
        plt.gca().add_patch(circle_i3); plt.gca().add_patch(circle_i3b)
        
        circle_h4 = plt.Circle((h4x,h4y), node_r,fill=True,facecolor = 'red',alpha=0.5,edgecolor = 'black',lw=2,zorder=100); plt.annotate(r'$H_4$',(h4x+dx,h4y+dy),zorder=110)
        circle_h4o = plt.Circle((h4x,h4y), node_r,fill=False,alpha=1.0,edgecolor = 'black',lw=2,zorder=105) 
        circle_h4b = plt.Circle((h4x,h4y), node_r*1.5, fill=True, facecolor = 'white',edgecolor = None,lw=1,zorder=10)
        plt.gca().add_patch(circle_h4); plt.gca().add_patch(circle_h4b); plt.gca().add_patch(circle_h4o)
    
        circle_h5 = plt.Circle((h5x,h5y), node_r, fill=True,facecolor = 'blue',alpha=0.5,edgecolor = 'black',lw=2,zorder=100); plt.annotate(r'$H_5$',(h5x+dx,h5y+dy),zorder=110)
        circle_h5o = plt.Circle((h5x,h5y), node_r, fill=False,alpha=1.0,edgecolor = 'black',lw=2,zorder=105) 
        circle_h5b = plt.Circle((h5x,h5y), node_r*1.5, fill=True, facecolor = 'white',edgecolor = None,lw=1,zorder=10);
        plt.gca().add_patch(circle_h5); plt.gca().add_patch(circle_h5b); plt.gca().add_patch(circle_h5o)
    
        circle_o6 = plt.Circle((o6x,o6y), node_r,fill=True,facecolor = 'orange',alpha=0.5,edgecolor = 'black',lw=2,zorder=100); plt.annotate(r'$O_6$',(o6x+dx,o6y+dy),zorder=110)
        circle_o6o = plt.Circle((o6x,o6y), node_r,fill=False,alpha=1.0,edgecolor = 'black',lw=2,zorder=105)
        circle_o6b = plt.Circle((o6x,o6y), node_r*1.5, fill=True, facecolor = 'white',edgecolor = None,lw=1,zorder=10);
        plt.gca().add_patch(circle_o6); plt.gca().add_patch(circle_o6b); plt.gca().add_patch(circle_o6o)
        
    plt.plot([i1x-edge,i1x],[i1y,i1y],color='grey',lw=1.0,zorder=1)
    plt.plot([i2x-edge,i2x],[i2y,i2y],color='grey',lw=1.0,zorder=1)
    plt.plot([i3x-edge,i3x],[i3y,i3y],color='grey',lw=1.0,zorder=1)
    
    plt.annotate(r'$x_1$ = ' + str(np.round(x1,2)),(i1x-buffer-1.6,i1y-0.05),size=8,zorder=200,color='grey',
                 bbox=dict(boxstyle="round,pad=0.0", edgecolor='white', facecolor='white', alpha=1.0),rotation=0,ha='left') 
    plt.annotate(r'$x_2$ = ' + str(np.round(x2,2)),(i2x-buffer-1.6,i2y-0.05),size=8,zorder=200,color='grey',
                 bbox=dict(boxstyle="round,pad=0.0", edgecolor='white', facecolor='white', alpha=1.0),rotation=0) 
    plt.annotate(r'$x_3$ = ' + str(np.round(x3,2)),(i3x-buffer-1.6,i3y-0.05),size=8,zorder=200,color='grey',
                 bbox=dict(boxstyle="round,pad=0.0", edgecolor='white', facecolor='white', alpha=1.0),rotation=0) 
    
    min_wt = np.min(np.abs([w14[iepoch],w24[iepoch],w34[iepoch],w15[iepoch],w25[iepoch],w35[iepoch],w46[iepoch],w56[iepoch]]))
    max_wt = np.max(np.abs([w14[iepoch],w24[iepoch],w34[iepoch],w15[iepoch],w25[iepoch],w35[iepoch],w46[iepoch],w56[iepoch]]))
  
    min_dwt = np.min(np.abs([dw14[iepoch],dw24[iepoch],dw34[iepoch],dw15[iepoch],dw25[iepoch],dw35[iepoch],dw46[iepoch],dw56[iepoch]]))
    max_dwt = np.max(np.abs([dw14[iepoch],dw24[iepoch],dw34[iepoch],dw15[iepoch],dw25[iepoch],dw35[iepoch],dw46[iepoch],dw56[iepoch]]))
 
    if (weights == True and forward == False and back == False):
        plt.plot([i1x,h4x],[i1y,h4y],color='orangered',alpha = alpha*abs(w14[iepoch]/max_wt)+min_alpha,lw=lw*abs(w14[iepoch]/max_wt)+min_lw,zorder=1)
        plt.plot([i2x,h4x],[i2y,h4y],color='red',alpha = alpha*abs(w24[iepoch]/max_wt)+min_alpha,lw=lw*abs(w24[iepoch]/max_wt)+min_lw,zorder=1)
        plt.plot([i3x,h4x],[i3y,h4y],color='darkred',alpha = alpha*abs(w34[iepoch]/max_wt)+min_alpha,lw=lw*abs(w34[iepoch]/max_wt)+min_lw,zorder=1)
    
        plt.plot([i1x,h5x],[i1y,h5y],color='dodgerblue',alpha = alpha*abs(w15[iepoch]/max_wt)+min_alpha,lw=lw*abs(w15[iepoch]/max_wt)+min_lw,zorder=1)
        plt.plot([i2x,h5x],[i2y,h5y],color='blue',alpha = alpha*abs(w25[iepoch]/max_wt)+min_alpha,lw=lw*abs(w25[iepoch]/max_wt)+min_lw,zorder=1)
        plt.plot([i3x,h5x],[i3y,h5y],color='darkblue',alpha = alpha*abs(w35[iepoch]/max_wt)+min_alpha,lw=lw*abs(w35[iepoch]/max_wt)+min_lw,zorder=1)
    
        plt.plot([h4x,o6x],[h4y,o6y],color='orange',alpha = alpha*abs(w46[iepoch]/max_wt)+min_alpha,lw=lw*abs(w46[iepoch]/max_wt)+min_lw,zorder=1)
        plt.plot([h5x,o6x],[h5y,o6y],color='darkorange',alpha = alpha*abs(w56[iepoch]/max_wt)+min_alpha,lw=lw*abs(w56[iepoch]/max_wt)+min_lw,zorder=1)
 
    if (weights == False and forward == True and back == False):
        plt.plot([i1x,h4x],[i1y,h4y],color='orangered',alpha = alpha*abs(x1*w14[iepoch]/max_signal)+min_alpha,lw=lw*abs(x1*w14[iepoch]/max_signal)+min_lw,zorder=1)
        plt.plot([i2x,h4x],[i2y,h4y],color='red',alpha = alpha*abs(x2*w24[iepoch]/max_signal)+min_alpha,lw=lw*abs(x2*w24[iepoch]/max_signal)+min_lw,zorder=1)
        plt.plot([i3x,h4x],[i3y,h4y],color='darkred',alpha = alpha*abs(x3*w34[iepoch]/max_signal)+min_alpha,lw=lw*abs(x3*w34[iepoch]/max_signal)+min_lw,zorder=1)
    
        plt.plot([i1x,h5x],[i1y,h5y],color='dodgerblue',alpha = alpha*abs(x1*w15[iepoch]/max_signal)+min_alpha,lw=lw*abs(x1*w15[iepoch]/max_signal)+min_lw,zorder=1)
        plt.plot([i2x,h5x],[i2y,h5y],color='blue',alpha = alpha*abs(x2*w25[iepoch]/max_signal)+min_alpha,lw=lw*abs(x2*w25[iepoch]/max_signal)+min_lw,zorder=1)
        plt.plot([i3x,h5x],[i3y,h5y],color='darkblue',alpha = alpha*abs(x3*w35[iepoch]/max_signal)+min_alpha,lw=lw*abs(x3*w35[iepoch]/max_signal)+min_lw,zorder=1)
    
        plt.plot([h4x,o6x],[h4y,o6y],color='orange',alpha = alpha*abs(y4[iepoch]*w46[iepoch]/max_signal)+min_alpha,lw=lw*abs(y4[iepoch]*w46[iepoch]/max_signal)+min_lw,zorder=1)
        plt.plot([h5x,o6x],[h5y,o6y],color='darkorange',alpha = alpha*abs(y5[iepoch]*w56[iepoch]/max_signal)+min_alpha,lw=lw*abs(y5[iepoch]*w56[iepoch]/max_signal)+min_lw,zorder=1)

    if (weights == False and forward == False and back == True):

        plt.plot([i1x,h4x],[i1y,h4y],color='orangered',alpha = alpha*abs(dw14[iepoch]/max_dwt)+min_alpha,lw=lw*abs(dw14[iepoch]/max_dwt)+min_lw,zorder=1)
        plt.plot([i2x,h4x],[i2y,h4y],color='red',alpha = alpha*abs(dw24[iepoch]/max_dwt)+min_alpha,lw=lw*abs(dw24[iepoch]/max_dwt)+min_lw,zorder=1)
        plt.plot([i3x,h4x],[i3y,h4y],color='darkred',alpha = alpha*abs(dw34[iepoch]/max_dwt)+min_alpha,lw=lw*abs(dw34[iepoch]/max_dwt)+min_lw,zorder=1)
    
        plt.plot([i1x,h5x],[i1y,h5y],color='dodgerblue',alpha = alpha*abs(dw15[iepoch]/max_dwt)+min_alpha,lw=lw*abs(dw15[iepoch]/max_dwt)+min_lw,zorder=1)
        plt.plot([i2x,h5x],[i2y,h5y],color='blue',alpha = alpha*abs(dw25[iepoch]/max_dwt)+min_alpha,lw=lw*abs(dw25[iepoch]/max_dwt)+min_lw,zorder=1)
        plt.plot([i3x,h5x],[i3y,h5y],color='darkblue',alpha = alpha*abs(dw35[iepoch]/max_dwt)+min_alpha,lw=lw*abs(dw35[iepoch]/max_dwt)+min_lw,zorder=1)
    
        plt.plot([h4x,o6x],[h4y,o6y],color='orange',alpha = alpha*abs(dw46[iepoch]/max_dwt)+min_alpha,lw=lw*abs(dw46[iepoch]/max_dwt)+min_lw,zorder=1)
        plt.plot([h5x,o6x],[h5y,o6y],color='darkorange',alpha = alpha*abs(dw56[iepoch]/max_dwt)+min_alpha,lw=lw*abs(dw56[iepoch]/max_dwt)+min_lw,zorder=1)
    
    if (weights == True and forward == True and back == True) or (weights == False and forward == False and back == False):
    
        plt.plot([i1x,h4x],[i1y,h4y],color='orangered',lw=lw+min_lw,zorder=1)
        plt.plot([i2x,h4x],[i2y,h4y],color='red',lw=lw+min_lw,zorder=1)
        plt.plot([i3x,h4x],[i3y,h4y],color='darkred',lw=lw+min_lw,zorder=1)
    
        plt.plot([i1x,h5x],[i1y,h5y],color='dodgerblue',lw=lw+min_lw,zorder=1)
        plt.plot([i2x,h5x],[i2y,h5y],color='blue',lw=lw+min_lw,zorder=1)
        plt.plot([i3x,h5x],[i3y,h5y],color='darkblue',lw=lw+min_lw,zorder=1)
    
        plt.plot([h4x,o6x],[h4y,o6y],color='orange',lw=lw+min_lw,zorder=1)
        plt.plot([h5x,o6x],[h5y,o6y],color='darkorange',lw=lw+min_lw,zorder=1)
    
    if forward == True:
    
        plt.plot(offsetx(i1x,2,-20),offsety(i1y,2,-20)+0.1,color='orangered',lw=1.0,zorder=1)
        plt.plot(offset_arrx(i1x,2,-20,0.2),offset_arry(i1y,2,-20,0.2)+0.1,color='orangered',lw=1.0,zorder=1)
        plt.annotate(r'$I_{1}$ = ' + str(np.round(x1,2)),(lintx(i1x,i1y,h4x,h4y,0.1),linty(i1x,i1y,h4x,h4y,0.1)),size=8,zorder=200,color='orangered',
                 bbox=dict(boxstyle="round,pad=0.0", edgecolor='white', facecolor='white', alpha=1.0),rotation=-20)
    
        plt.plot(offsetx(i2x,2,12),offsety(i2y,2,12)+0.1,color='red',lw=1.0,zorder=1)
        plt.plot(offset_arrx(i2x,2,12,0.2),offset_arry(i2y,2,12,0.2)+0.1,color='red',lw=1.0,zorder=1)
        plt.annotate(r'$I_{2}$ = ' + str(np.round(x2,2)),(lintx(i2x,i2y,h4x,h4y,0.1),linty(i2x,i2y,h4x,h4y,0.1)+0.24),size=8,zorder=200,color='red',
                 bbox=dict(boxstyle="round,pad=0.0", edgecolor='white', facecolor='white', alpha=1.0),rotation=12)
    
        plt.plot(offsetx(i3x,2,38),offsety(i3y,2,38)+0.1,color='darkred',lw=1.0,zorder=1)
        plt.plot(offset_arrx(i3x,2,38,0.2),offset_arry(i3y,2,38,0.2)+0.1,color='darkred',lw=1.0,zorder=1)
        plt.annotate(r'$I_{3}$ = ' + str(np.round(x3,2)),(lintx(i3x,i3y,h4x,h4y,0.08)-0.2,linty(i3x,i3y,h4x,h4y,0.08)+0.2),size=8,zorder=200,color='darkred',
                 bbox=dict(boxstyle="round,pad=0.0", edgecolor='white', facecolor='white', alpha=1.0),rotation=38)
    
        plt.plot(offsetx(h4x,2,-12),offsety(h4y,2,-12)+0.1,color='orange',lw=1.0,zorder=1)
        plt.plot(offset_arrx(h4x,2,-12,0.2),offset_arry(h4y,2,-12,0.2)+0.1,color='orange',lw=1.0,zorder=1)
        plt.annotate(r'$H_{4}$ = ' + str(np.round(y4[iepoch],2)),(lintx(h4x,h4y,o6x,o6y,0.08),linty(h4x,h4y,o6x,o6y,0.08)-0.0),size=8,zorder=200,color='orange',
                 bbox=dict(boxstyle="round,pad=0.0", edgecolor='white', facecolor='white', alpha=1.0),rotation=-12)
        
        plt.annotate(r'$H_{4_{in}}$ = ' + str(np.round(y4in[iepoch],2)),(lintx(h4x,h4y,o6x,o6y,0.08)-0.5,linty(h4x,h4y,o6x,o6y,0.08)+0.5),size=8,zorder=200,color='orange',
                 bbox=dict(boxstyle="round,pad=0.0", edgecolor='white', facecolor='white', alpha=1.0),rotation=-12)
    
        plt.plot(offsetx(h5x,2,12),offsety(h5y,2,12)+0.1,color='darkorange',lw=1.0,zorder=1)
        plt.plot(offset_arrx(h5x,2,12,0.2),offset_arry(h5y,2,12,0.2)+0.1,color='darkorange',lw=1.0,zorder=1)
        plt.annotate(r'$H_{5}$ = ' + str(np.round(y5[iepoch],2)),(lintx(h5x,h5y,o6x,o6y,0.07),linty(h5x,h5y,o6x,o6y,0.07)+0.25),size=8,zorder=200,color='darkorange',
                 bbox=dict(boxstyle="round,pad=0.0", edgecolor='white', facecolor='white', alpha=1.0),rotation=12)

        plt.annotate(r'$H_{5_{in}}$ = ' + str(np.round(y5in[iepoch],2)),(lintx(h5x,h5y,o6x,o6y,0.08)-0.5,linty(h5x,h5y,o6x,o6y,0.08)+0.5),size=8,zorder=200,color='darkorange',
                 bbox=dict(boxstyle="round,pad=0.0", edgecolor='white', facecolor='white', alpha=1.0),rotation=12)
        
        plt.plot([o6x+edge,o6x],[o6y,o6y],color='grey',lw=1.0,zorder=1)
        plt.annotate(r'$\hat{y}$ = ' + str(np.round(y6[iepoch],2)),(o6x+buffer+0.7,o6y-0.05),size=8,zorder=300,color='grey',
                 bbox=dict(boxstyle="round,pad=0.0", edgecolor='white', facecolor='white', alpha=1.0),rotation=0) 

#        plt.annotate(r'$\frac{\partial P}{\partial \hat{y}}$ = ' + str(np.round(d6[iepoch],2)),(o6x,o6y-1.2),size=10,
#                bbox=dict(boxstyle="round,pad=0.0", edgecolor='white', facecolor='white', alpha=1.0),zorder=200)

        plt.annotate(r'$y$ = ' + str(np.round(y,2)),(o6x+buffer+0.7,o6y+0.5),size=8,zorder=300,color='grey',
                 bbox=dict(boxstyle="round,pad=0.0", edgecolor='white', facecolor='white', alpha=1.0),rotation=0) 

    plt.plot([h4x,h4x-0.5],[h4y,h4y+1.0],color='red',zorder=5)
    plt.annotate(r'$b_{4}$ = ' + str(np.round(b4[iepoch],2)),(h4x-0.7,h4y+1.2),size=8,zorder=200,color='red',
        bbox=dict(boxstyle="round,pad=0.0", edgecolor='white', facecolor='white', alpha=1.0),rotation=0)  
    
    plt.plot([h5x,h5x-0.5],[h5y,h5y+1.0],color='blue',zorder=5)
    plt.annotate(r'$b_{5}$ = ' + str(np.round(b5[iepoch],2)),(h5x-0.7,h5y+1.2),size=8,zorder=200,color='blue',
        bbox=dict(boxstyle="round,pad=0.0", edgecolor='white', facecolor='white', alpha=1.0),rotation=0)  
    
    plt.plot([o6x,o6x-0.5],[o6y,o6y+1.0],color='orange',zorder=5)
    plt.annotate(r'$b_{6}$ = ' + str(np.round(b6[iepoch],2)),(o6x-0.7,o6y+1.2),size=8,zorder=200,color='orange',
        bbox=dict(boxstyle="round,pad=0.0", edgecolor='white', facecolor='white', alpha=1.0),rotation=0)  
    
    plt.plot([o6x+edge,o6x],[o6y,o6y],color='grey',lw=1.0,zorder=1)
    plt.annotate(r'$y$ = ' + str(np.round(y,2)),(o6x+buffer+0.7,o6y-0.05),size=8,zorder=200,color='grey',
        bbox=dict(boxstyle="round,pad=0.0", edgecolor='white', facecolor='white', alpha=1.0),rotation=0) 
    
    if back == True:
    
        plt.annotate(r'$\frac{\partial P}{\partial O_{6_{in}}}$ = ' + str(np.round(d6[iepoch],2)),(o6x-0.5,o6y-1.2),size=10,zorder=100)
        plt.annotate(r'$\frac{\partial P}{\partial H_{4_{in}}}$ = ' + str(np.round(d4[iepoch],2)),(h4x-0.5,h4y-0.7),size=10,zorder=100)
        plt.annotate(r'$\frac{\partial P}{\partial H_{5_{in}}}$ = ' + str(np.round(d5[iepoch],2)),(h5x-0.5,h5y-0.7),size=10,zorder=100)
    
        plt.annotate(r'$\frac{\partial P}{\partial \hat{y}}$ = ' + str(np.round(d6[iepoch],2)),(o6x,o6y-1.8),size=10,zorder=100)
        plt.annotate(r'$\frac{\partial P}{\partial H_{4_{out}}}$ = ' + str(np.round(w46[iepoch]*d6[iepoch],2)),(h4x,h4y-1.2),size=10,zorder=100)
        plt.annotate(r'$\frac{\partial P}{\partial H_{5_{out}}}$ = ' + str(np.round(w56[iepoch]*d6[iepoch],2)),(h5x,h5y-1.2),size=10,zorder=100)
   
        plt.annotate(r'$\frac{\partial P}{\partial X_1}$ = ' + r'${0:s}$'.format(as_si(d1[iepoch],2)),(i1x-2.0,i1y-0.9),size=10,zorder=100)
        plt.annotate(r'$\frac{\partial P}{\partial X_2}$ = ' + r'${0:s}$'.format(as_si(d2[iepoch],2)),(i2x-2.0,i2y-0.9),size=10,zorder=100)
        plt.annotate(r'$\frac{\partial P}{\partial X_3}$ = ' + r'${0:s}$'.format(as_si(d3[iepoch],2)),(i3x-2.0,i3y-0.9),size=10,zorder=100)
        
        plt.annotate(r'$\frac{\partial P}{\partial I_1}$ = ' + r'${0:s}$'.format(as_si(d1[iepoch],2)),(i1x-1.5,i1y-1.4),size=10,zorder=100)
        plt.annotate(r'$\frac{\partial P}{\partial I_2}$ = ' + r'${0:s}$'.format(as_si(d2[iepoch],2)),(i2x-1.5,i2y-1.4),size=10,zorder=100)
        plt.annotate(r'$\frac{\partial P}{\partial I_3}$ = ' + r'${0:s}$'.format(as_si(d3[iepoch],2)),(i3x-1.5,i3y-1.4),size=10,zorder=100)

        plt.plot(lint_intx(h4x, h4y, o6x, o6y,0.4,0.6),lint_inty(h4x,h4y,o6x,o6y,0.4,0.6)-0.1,color='orange',lw=1.0,zorder=1)
        plt.plot(lint_int_arrx(o6x,o6y,h4x,h4y,0.4,0.6,0.2),lint_int_arry(o6x,o6y,h4x,h4y,0.4,0.6,0.2)-0.1,color='orange',lw=1.0,zorder=1)
        plt.annotate(r'$\frac{\partial P}{\partial \lambda_{4,6}} =$' + r'${0:s}$'.format(as_si(dw46[iepoch]/lr,2)),(lintx(h4x,h4y,o6x,o6y,0.5)-0.6,linty(h4x,h4y,o6x,o6y,0.5)-0.72),size=7,zorder=200,color='orange',
                  bbox=dict(boxstyle="round,pad=0.0", edgecolor='white', facecolor='white', alpha=1.0),rotation=-11)
    
        plt.plot(lint_intx(h5x, h5y, o6x, o6y,0.4,0.6),lint_inty(h5x,h5y,o6x,o6y,0.4,0.6)-0.1,color='darkorange',lw=1.0,zorder=1)
        plt.plot(lint_int_arrx(o6x,o6y,h5x,h5y,0.4,0.6,0.2),lint_int_arry(o6x,o6y,h5x,h5y,0.4,0.6,0.2)-0.1,color='darkorange',lw=1.0,zorder=1)
        plt.annotate(r'$\frac{\partial P}{\partial \lambda_{5,6}} =$' + r'${0:s}$'.format(as_si(dw56[iepoch]/lr,2)),(lintx(h5x,h5y,o6x,o6y,0.5)-0.4,linty(h5x,h5y,o6x,o6y,0.5)-0.6),size=7,zorder=200,color='darkorange',
                  bbox=dict(boxstyle="round,pad=0.0", edgecolor='white', facecolor='white', alpha=1.0),rotation=12)

        plt.plot(lint_intx(i1x, i1y, h4x, h4y,0.4,0.6),lint_inty(i1x,i1y,h4x,h4y,0.4,0.6)-0.1,color='orangered',lw=1.0,zorder=1)
        plt.plot(lint_int_arrx(h4x,h4y,i1x,i1y,0.4,0.6,0.2),lint_int_arry(h4x,h4y,i1x,i1y,0.4,0.6,0.2)-0.1,color='orangered',lw=1.0,zorder=1)
        plt.annotate(r'$\frac{\partial P}{\partial \lambda_{1,4}} =$' + r'${0:s}$'.format(as_si(dw14[iepoch]/lr,2)),(lintx(i1x,i1y,h4x,h4y,0.5)-0.6,linty(i1x,i1y,h4x,h4y,0.5)-1.0),size=7,zorder=200,color='orangered',
                  bbox=dict(boxstyle="round,pad=0.0", edgecolor='white', facecolor='white', alpha=1.0),rotation=-20)
    
        plt.plot(lint_intx(i2x, i2y, h4x, h4y,0.3,0.5),lint_inty(i2x,i2y,h4x,h4y,0.3,0.5)-0.1,color='red',lw=1.0,zorder=1)
        plt.plot(lint_int_arrx(h4x,h4y,i2x,i2y,0.5,0.7,0.2),lint_int_arry(h4x,h4y,i2x,i2y,0.5,0.7,0.2)-0.12,color='red',lw=1.0,zorder=1)
        plt.annotate(r'$\frac{\partial P}{\partial \lambda_{2,4}} =$' + r'${0:s}$'.format(as_si(dw24[iepoch]/lr,2)),(lintx(i2x,i2y,h4x,h4y,0.5)-1.05,linty(i2x,i2y,h4x,h4y,0.5)-0.7),size=7,zorder=200,color='red',
                  bbox=dict(boxstyle="round,pad=0.0", edgecolor='white', facecolor='white', alpha=1.0),rotation=12)
    
        plt.plot(lint_intx(i3x, i3y, h4x, h4y,0.2,0.4),lint_inty(i3x,i3y,h4x,h4y,0.2,0.4)-0.1,color='darkred',lw=1.0,zorder=1)
        plt.plot(lint_int_arrx(h4x,h4y,i3x,i3y,0.5,0.8,0.2),lint_int_arry(h4x,h4y,i3x,i3y,0.5,0.8,0.2)-0.12,color='darkred',lw=1.0,zorder=1)
        plt.annotate(r'$\frac{\partial P}{\partial \lambda_{3,4}} =$' + r'${0:s}$'.format(as_si(dw34[iepoch]/lr,2)),(lintx(i3x,i3y,h4x,h4y,0.5)-1.7,linty(i3x,i3y,h4x,h4y,0.5)-1.7),size=7,zorder=200,color='darkred',
                  bbox=dict(boxstyle="round,pad=0.0", edgecolor='white', facecolor='white', alpha=1.0),rotation=38)
    
        plt.plot(lint_intx(i3x, i3y, h5x, h5y,0.4,0.6),lint_inty(i3x,i3y,h5x,h5y,0.4,0.6)-0.1,color='darkblue',lw=1.0,zorder=1)
        plt.plot(lint_int_arrx(h5x,h5y,i3x,i3y,0.4,0.6,0.2),lint_int_arry(h5x,h5y,i3x,i3y,0.4,0.6,0.2)-0.12,color='darkblue',lw=1.0,zorder=1)
        plt.annotate(r'$\frac{\partial P}{\partial \lambda_{3,5}} =$' + r'${0:s}$'.format(as_si(dw35[iepoch]/lr,2)),(lintx(i3x,i3y,h5x,h5y,0.5)-0.4,linty(i3x,i3y,h5x,h5y,0.5)-0.6),size=7,zorder=200,color='darkblue',
                  bbox=dict(boxstyle="round,pad=0.0", edgecolor='white', facecolor='white', alpha=1.0),rotation=20)
    
        plt.plot(lint_intx(i2x, i2y, h5x, h5y,0.3,0.5),lint_inty(i2x,i2y,h5x,h5y,0.3,0.5)-0.1,color='blue',lw=1.0,zorder=1)
        plt.plot(lint_int_arrx(h5x,h5y,i2x,i2y,0.3,0.7,0.2),lint_int_arry(h5x,h5y,i2x,i2y,0.3,0.7,0.2)-0.12,color='blue',lw=1.0,zorder=1)
        plt.annotate(r'$\frac{\partial P}{\partial \lambda_{2,5}} =$' + r'${0:s}$'.format(as_si(dw25[iepoch]/lr,2)),(lintx(i2x,i2y,h5x,h5y,0.5)-1.2,linty(i2x,i2y,h5x,h5y,0.5)-0.65),size=7,zorder=200,color='blue',
                  bbox=dict(boxstyle="round,pad=0.0", edgecolor='white', facecolor='white', alpha=1.0),rotation=-12)
    
        plt.plot(lint_intx(i1x, i1y, h5x, h5y,0.2,0.4),lint_inty(i1x,i1y,h5x,h5y,0.2,0.4)-0.1,color='dodgerblue',lw=1.0,zorder=1)
        plt.plot(lint_int_arrx(h5x,h5y,i1x,i1y,0.2,0.8,0.2),lint_int_arry(h5x,h5y,i1x,i1y,0.2,0.8,0.2)-0.12,color='dodgerblue',lw=1.0,zorder=1)
        plt.annotate(r'$\frac{\partial P}{\partial \lambda_{1,5}} =$' + r'${0:s}$'.format(as_si(dw15[iepoch]/lr,2)),(lintx(i1x,i1y,h5x,h5y,0.5)-2.2,linty(i1x,i1y,h4x,h4y,0.5)-0.2),size=7,zorder=200,color='dodgerblue',
                  bbox=dict(boxstyle="round,pad=0.0", edgecolor='white', facecolor='white', alpha=1.0),rotation=-36,xycoords = 'data',va="top",ha="left")
    
    if forward == False:
    
        plt.annotate(r'$\lambda_{1,4}$ = ' + str(np.round(w14[iepoch],2)),((i1x+h4x)*0.45,(i1y+h4y)*0.5+0.7),size=8,zorder=200,color='orangered',
                     bbox=dict(boxstyle="round,pad=0.0", edgecolor='white', facecolor='white', alpha=1.0),rotation=-18,xycoords = 'data',va="top",ha="left") 
        plt.annotate(r'$\lambda_{2,4}$ = ' + str(np.round(w24[iepoch],2)),((i2x+h4x)*0.45-0.3,(i2y+h4y)*0.5-0.03),size=8,zorder=200,color='red',
                     bbox=dict(boxstyle="round,pad=0.0", edgecolor='white', facecolor='white', alpha=1.0),rotation=12) 
        plt.annotate(r'$\lambda_{3,4}$ = ' + str(np.round(w34[iepoch],2)),((i3x+h4x)*0.45-1.2,(i3y+h4y)*0.5-1.1),size=8,zorder=200,color='darkred',
                     bbox=dict(boxstyle="round,pad=0.0", edgecolor='white', facecolor='white', alpha=1.0),rotation=38) 
        
        plt.annotate(r'$\lambda_{1,5}$ = ' + str(np.round(w15[iepoch],2)),((i1x+h5x)*0.55-2.5,(i1y+h5y)*0.5+0.7),size=8,zorder=200,color='dodgerblue',
            bbox=dict(boxstyle="round,pad=0.0", edgecolor='white', facecolor='white', alpha=1.0),rotation=-36) 
        plt.annotate(r'$\lambda_{2,5}$ = ' + str(np.round(w25[iepoch],2)),((i2x+h5x)*0.55-1.5,(i2y+h5y)*0.5+0.05),size=8,zorder=200,color='blue',
                     bbox=dict(boxstyle="round,pad=0.0", edgecolor='white', facecolor='white', alpha=1.0),rotation=-12) 
        plt.annotate(r'$\lambda_{3,5}$ = ' + str(np.round(w35[iepoch],2)),((i3x+h5x)*0.55-1.0,(i3y+h5y)*0.5+0.1),size=8,zorder=200,color='darkblue',
                     bbox=dict(boxstyle="round,pad=0.0", edgecolor='white', facecolor='white', alpha=1.0),rotation=18) 
        
        plt.annotate(r'$\lambda_{4,6}$ = ' + str(np.round(w46[iepoch],2)),((h4x+o6x)*0.47,(h4y+o6y)*0.47+0.39),size=8,zorder=200,color='orange',
            bbox=dict(boxstyle="round,pad=0.0", edgecolor='white', facecolor='white', alpha=1.0),rotation=-12) 
        plt.annotate(r'$\lambda_{5,6}$ = ' + str(np.round(w56[iepoch],2)),((h5x+o6x)*0.47,(h5y+o6y)*0.47+0.26),size=8,zorder=200,color='darkorange',
            bbox=dict(boxstyle="round,pad=0.0", edgecolor='white', facecolor='white', alpha=1.0),rotation=12) 
        
    if forward == True:
     
        plt.annotate(r'$\lambda_{1,4} \times X_1$ = ' + str(np.round(x1*w14[iepoch],2)),((i1x+h4x)*0.45,(i1y+h4y)*0.5+0.70),size=8,zorder=200,color='orangered',
                     bbox=dict(boxstyle="round,pad=0.0", edgecolor='white', facecolor='white', alpha=1.0),rotation=-18,xycoords = 'data',va="top",ha="left") 
        plt.annotate(r'$\lambda_{2,4} \times X_2$ = ' + str(np.round(x2*w24[iepoch],2)),((i2x+h4x)*0.45-0.3,(i2y+h4y)*0.5+0.70),size=8,zorder=200,color='red',
                     bbox=dict(boxstyle="round,pad=0.0", edgecolor='white', facecolor='white', alpha=1.0),rotation=10,xycoords = 'data',va="top",ha="left") 
        plt.annotate(r'$\lambda_{3,4} \times X_3$ = ' + str(np.round(x3*w34[iepoch],2)),((i3x+h4x)*0.45-1.4,(i3y+h4y)*0.5-1.3),size=8,zorder=200,color='darkred',
                     bbox=dict(boxstyle="round,pad=0.0", edgecolor='white', facecolor='white', alpha=1.0),rotation=36,xycoords = 'data',va="bottom",ha="left") 
        
        plt.annotate(r'$\lambda_{1,5} \times X_1$ = ' + str(np.round(x1*w15[iepoch],2)),((i1x+h5x)*0.55-2.5,(i1y+h5y)*0.5+1.9),size=8,zorder=200,color='dodgerblue',
            bbox=dict(boxstyle="round,pad=0.0", edgecolor='white', facecolor='white', alpha=1.0),rotation=-36,xycoords = 'data',va="top",ha="left") 
        plt.annotate(r'$\lambda_{2,5} \times X_2$ = ' + str(np.round(x2*w25[iepoch],2)),((i2x+h5x)*0.55-1.5,(i2y+h5y)*0.5+0.7),size=8,zorder=200,color='blue',
                     bbox=dict(boxstyle="round,pad=0.0", edgecolor='white', facecolor='white', alpha=1.0),rotation=-12,xycoords = 'data',va="top",ha="left") 
        plt.annotate(r'$\lambda_{3,5} \times X_3$ = ' + str(np.round(x3*w35[iepoch],2)),((i3x+h5x)*0.55-1.0,(i3y+h5y)*0.5+1.1),size=8,zorder=200,color='darkblue',
                     bbox=dict(boxstyle="round,pad=0.0", edgecolor='white', facecolor='white', alpha=1.0),rotation=18,xycoords = 'data',va="top",ha="left") 
        
        plt.annotate(r'$\lambda_{4,6} \times Y_4$ = ' + str(np.round(y4[iepoch]*w46[iepoch],2)),((h4x+o6x)*0.47,(h4y+o6y)*0.47+1.0),size=8,zorder=200,color='orange',
            bbox=dict(boxstyle="round,pad=0.0", edgecolor='white', facecolor='white', alpha=1.0),rotation=-12,xycoords = 'data',va="top",ha="left") 
        plt.annotate(r'$\lambda_{5,6} \times Y_5$ = ' + str(np.round(y5[iepoch]*w56[iepoch],2)),((h5x+o6x)*0.47,(h5y+o6y)*0.47+1.0),size=8,zorder=200,color='darkorange',
            bbox=dict(boxstyle="round,pad=0.0", edgecolor='white', facecolor='white', alpha=1.0),rotation=12,xycoords = 'data',va="top",ha="left") 
    

    
    plt.plot([0.5,20,20,0.5,0.5],[-1,-1,10,10,-1],color='black')
    #plt.scatter(0,10,color='yellow')
    
    plt.subplots_adjust(left=0.0, bottom=0.0, right=1.5, top=1.0, wspace=0.2, hspace=0.2); plt.show()
       
# connect the function to make the samples and plot to the widgets    
interactive_plot = widgets.interactive_output(run_plot, {'nepoch':nepoch,'lr':lr,'weights':weights,'forward':forward,'back':back,'seed':seed})
interactive_plot.clear_output(wait = True)               # reduce flickering by delaying plot updating
