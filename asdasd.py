-(
    a1 = ((-210*torch.cos(2*c)-75)*d**7+(-1260*torch.cos(2*c)-378)*d**5)*torch.log(torch.sqrt(1-d**2)+1)
    a2 = +(75*d**7+378*d**5)*torch.log(torch.sqrt(1-d**2)-1)
    a3 = +(210*torch.cos(2*c)*d**7+1260*torch.cos(2*c)*d**5)*torch.log(d)
    a4 = -24*torch.arccos(d)
    a5 = +75.j*torch.arctan2(0,torch.sqrt(1-dr**2)-1)*d**7
    a6  = +torch.sqrt(1-d**2)*((1134*torch.cos(2*c)+746)*d**5+(392*torch.cos(2*c)+152)*d**3+(32-56*torch.cos(2*c))*d)
    a7 = +378.j*torch.arctan2(0,torch.sqrt(1-dr**2)-1)*d**5
    term = -(a1 + a2 + a3 + a4 + a5 + a6 + a7) /(24*torch.pi)
)/24*torch.pi

(
    a1 = ((75-210*torch.cos(2*c))*d**7+(378-1260*torch.cos(2*c))*d**5)*torch.log(torch.sqrt(1-d**2)+1)
    a2 = +(-75*d**7-378*d**5)*torch.log(torch.sqrt(1-d**2)-1)
    a3 = +(210*torch.cos(2*c)*d**7+1260*torch.cos(2*c)*d**5)*torch.log(d)
    a4 = +24*torch.arccos(d)
    a5 = -75.j*torch.arctan2(0,torch.sqrt(1-dr**2)-1)*d**7
    a6 = +torch.sqrt(1-d**2)*((1134*torch.cos(2*c)-746)*d**5+(392*torch.cos(2*c)-152)*d**3+(-56*torch.cos(2*c)-32)*d)
    a7 = -378.j*torch.arctan2(0,torch.sqrt(1-dr**2)-1)*d**5
    )/24*torch.pi

-(
    a1 = (-105*torch.sin(2*c)*d**7-630*torch.sin(2*c)*d**5)*torch.log(torch.sqrt(1-d**2)+1)
    a2 = +(105*torch.sin(2*c)*d**7+630*torch.sin(2*c)*d**5)*torch.log(d)
    a3 = +torch.sqrt(d**2-1)*(567.j*torch.sin(2*c)*d**5+196.j*torch.sin(2*c)*d**3-28.j*torch.sin(2*c)*d)
)/12*torch.pi