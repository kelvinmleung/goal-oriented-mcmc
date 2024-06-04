using Images
using FileIO

using Plots
using Statistics
using LinearAlgebra
using Random, Distributions
using AOE

λ_ranges = [400.0 1300.0; 1450.0 1780.0; 2051.0 2451.0]
priormodel, wls = get_priormodel(:standard; λ_ranges) # PriorModel instance
rtmodel = AOE.get_radiative_transfer_modtran(:LUTRT1; λ_ranges);

function aoe_fwdfun(x)
    xa = x[1:2]
    xs = x[3:end]
    AOE.fwdfun(xa, xs, rtmodel) 
end

function aoe_gradfwdfun(x, fx)
    xa = x[1:2]
    xs = x[3:end]
    # AOE.gradfwd(xa, xs, rtmodel) 
    AOE.gradfwd_accel(xa, xs, rtmodel, fx) 
end


function plotImg(img, lbl)
    display(Plots.plot(img, axis = false, title=lbl, dpi=300))
end

function plotImgVec(imgvec, h, w, lbl)
    display(Plots.plot(Gray.(reshape(imgvec, (h, w))), axis=false, title=lbl, dpi=300))
end

function extractGoal(imgvec, h, w, ycoord, xcoord)
    img = reshape(imgvec, (w,h))
    img[xcoord, ycoord][:]
end


function goalMatrix(hgt_comp, wdth_comp, sf)
    O = zeros(hgt_comp*wdth_comp, numPix)    
    for i in 1:hgt_comp
        for j in 1:wdth_comp
            rowtemp = zeros(hgt, wdth)
            rowtemp[(i-1)*sf+1:i*sf, (j-1)*sf+1:j*sf] .= 1
            O[(j-1)*hgt_comp + i, :] = reshape(rowtemp, numPix) / sf^2 
        end
    end
    return O
end

function goalMatrix_selection(hgt, wdth, hgt_range, wdth_range)
    hgt_comp, wdth_comp = length(hgt_range), length(wdth_range)

    O = zeros(hgt_comp * wdth_comp, hgt * wdth)    
    for i in 1:hgt_comp
        for j in 1:wdth_comp
            # O[(i-1)*wdth_comp + j, (hgt_range[i]-1)*wdth + wdth_range[j]] = 1
            O[(i-1)*wdth_comp + j, (hgt_range[i]-1)*wdth + wdth_range[j]] = 1
        end
    end
    return O
end

function construct_prior(hgt, wdth, param_l)
    n = hgt*wdth
    xcoord, ycoord = repeat(collect(1:wdth),1,hgt)', repeat(collect(1:hgt),1,wdth)
    x1, x2 = repeat(xcoord[:], 1, n), repeat(xcoord[:], 1, n)'
    y1, y2 = repeat(ycoord[:], 1, n), repeat(ycoord[:], 1, n)'
    dist = (x1 - x2).^2 + (y1 - y2).^2
    pr = exp.(-1 / (2 * param_l) * dist)
    return pr .* (pr .> 1e-10)
end


# function fwdtoy(x)
#     x1, x2, x3 = x
#     g1 = cos(x1 )+ x2^2 
#     g2 = x1 - x3 
#     # g3 = x1^2 + cos(x2)
#     g3 = x1^2 + sin(x2)
#     return [g1; g2; g3]
# end

# function dfwdtoy(x)
#     x1, x2, x3 = x
#     dg = [[-sin(x1) 2*x2 0]; 
#         [1 0 -1]; 
#         [2*x1 cos(x2) 0]]
#     return dg
# end

function fwdtoy(x)
    x1, x2 = x
    g1 = x1 + 10*x2
    g2 = x1 - x2
    return [g1; g2]
end

function dfwdtoy(x)
    x1, x2 = x
    dg = [[1 10]; 
        [1 -1]]
    return dg
end


function blurMatrix_linear(σ_k², img)
    hgt, wdth = size(img) 
    kernel(x,y) = exp(- 1/σ_k² * (x^2 + y^2))
    A = zeros(hgt, wdth,hgt, wdth)
    for i in CartesianIndices(A)
        A[i] = kernel(i[1]-i[3],i[2]-i[4])
    end
    l1 =  Int(round(hgt/2))
    l2 =  Int(round(wdth/2))
    sumkernal = sum(kernel(x,y) for x in -l1:l1, y in -l2:l2)

    display(sumkernal)
    G = reshape(A,length(img),length(img)) ./ sumkernal
end

function blurMatrix_nonlinear(σ_k², x, hgt, wdth)
    # hgt, wdth = size(img) 
    kernel(x,y) = exp(- 1/σ_k² * (x^2 + y^2)) 
    brightness = mean(x)
    A = zeros(hgt, wdth, hgt, wdth)
    for i in CartesianIndices(A)
        A[i] = kernel(i[1]-i[3],i[2]-i[4]) #* (img[i[1], i[2]] * img[i[3], i[4]] / brightness^2)
    end
    l1 =  Int(round(hgt/2))
    l2 =  Int(round(wdth/2))
    sumkernal = sum(kernel(x,y) for x in -l1:l1, y in -l2:l2)
    G = reshape(A,length(x),length(x)) ./ sumkernal
    return 2 * G * x.^2 , 4 * G * diagm(x)
end

function blurMatrix_linear2(σ_k², x, hgt, wdth)
    kernel(x,y) = exp(- 1/σ_k² * (x^2 + y^2))
    A = zeros(hgt, wdth,hgt, wdth)
    for i in CartesianIndices(A)
        A[i] = kernel(i[1]-i[3],i[2]-i[4])
    end
    l1 =  Int(round(hgt/2))
    l2 =  Int(round(wdth/2))
    sumkernal = sum(kernel(x,y) for x in -l1:l1, y in -l2:l2)
    G = reshape(A,length(img),length(img)) ./ sumkernal
    return G * x, G
end

function blurMatrix_nonlinear2(σ_1², σ_2², x, hgt, wdth; calc_der=true) # how to speed up this calculation??
    # hgt, wdth = size(img) 
    n = length(x)
    kernel(t1,t2,tx) = exp(-1/(σ_1²+ (σ_2²-σ_1²) * tx) * (t1^2 + t2^2)) 
    # dkernel(t1,t2,tx) =  (t1^2 + t2^2)  * (σ_2²-σ_1²)/ (σ_1²+ (σ_2²-σ_1²) * tx)^2 * kernel(t1,t2,tx) 
    A = zeros(hgt, wdth, hgt, wdth)
    if calc_der
        dA = zeros(hgt, wdth, hgt, wdth)
    end
    sumkernal = 0
    l1 =  Int(round(hgt/2))
    l2 =  Int(round(wdth/2))

    # @time begin
    sumkernal=zeros(n)
    for i in 1:n
        sumkernal[i] = sum(kernel(t1,t2, x[i]) for t1 in -l1:l1, t2 in -l2:l2)
    end
    # end
    
    # @time begin
    for i in CartesianIndices(A)
        xpixel = i[1] + (i[2]-1) * hgt
        if x[xpixel] > 1
            xinput = 1
        elseif x[xpixel] < 0
            xinput = 0
        else
            xinput = x[xpixel] 
        end

        A[i] = kernel(i[1]-i[3], i[2]-i[4], xinput) / sumkernal[xpixel]
        # dA[i] = dkernel(i[1]-i[3], i[2]-i[4], xinput)
        if calc_der
            dA[i] = (i[1]-i[3]^2 + i[2]-i[4]^2)  * (σ_2²-σ_1²)/ (σ_1²+ (σ_2²-σ_1²) * xinput)^2 * A[i]
        end
    end
    # end
    G = reshape(A,n,n) 
    if calc_der
        dAreshape = reshape(dA,n,n)
        return G * x, G + diagm(dAreshape * x)
    else
        return G*x
    end
end

function blurMatrix_nonlinear2_fast(σ_1², σ_2², x, hgt, wdth) 
    n = length(x)

    @time begin
    D = zeros(hgt, wdth, hgt, wdth)
    sumkernal = zeros(n)
    for i in CartesianIndices(D)
        D[i] = (i[1]-i[3])^2 +(i[2]-i[4])^2
    end
    end
    distmat = reshape(D,n,n)
    xmat = repeat(x,1,n)
    @time begin
    kernel(t1,t2,tx) = exp(-1/(σ_1²+ (σ_2²-σ_1²) * tx) * (t1^2 + t2^2)) 
    for i in 1:n
        sumkernal[i] = sum(kernel(t1,t2, x[i]) for t1 in -l1:l1, t2 in -l2:l2)
    end
    end
    @time begin
    G = exp.(-1 ./(σ_1² .+ (σ_2²-σ_1²) * xmat) .* distmat) #./ repeat(sumkernal,1,n)
    dG = (σ_2²-σ_1²) ./ (σ_1² .+ (σ_2²-σ_1²) * xmat).^2 .* distmat .* G
    end

    # THEN CHECK THE SPEED AND whether it's the same as blurMatrix
    return G * x, G + diagm(dG * x)
end

function blurMatrix_nonlinear2_multi(σ_1², σ_2², xmat, hgt, wdth) #### MAKE THIS
    # hgt, wdth = size(img) 
    n, m = size(xmat)
    kernel(t1,t2,tx_vec) = exp.(-1 ./(σ_1² .+ (σ_2²-σ_1²) * tx_vec) * (t1^2 + t2^2)) 
    dkernel(t1,t2,tx_vec) =  (t1^2 + t2^2)  * (σ_2²-σ_1²) ./ (σ_1² .+ (σ_2²-σ_1²) * tx_vec).^2 .* kernel(t1,t2,tx_vec) 
    A = zeros(hgt, wdth, hgt, wdth, m)
    dA = zeros(hgt, wdth, hgt, wdth, m)
    for i in CartesianIndices(zeros(hgt, wdth, hgt, wdth))
        xpixel = i[1] + (i[2]-1) * hgt
        xinput = xmat[xpixel,:]
        for j in 1:m
            if xmat[xpixel,j] > 1
                xinput[j] = 1
            elseif xmat[xpixel,j]< 0
                xinput[j] = 0
            end
        end
        A[i[1],i[2],i[3],i[4], :] = kernel(i[1]-i[3], i[2]-i[4], xinput)
        dA[i[1],i[2],i[3],i[4], :] = dkernel(i[1]-i[3], i[2]-i[4], xinput)
    end
    l1 =  Int(round(hgt/2))
    l2 =  Int(round(wdth/2))
    G = reshape(A,n,n,m)
    dAreshape = reshape(dA,n,n,m)
    for i in 1:n
        sumkernal = sum(kernel(t1,t2, xmat[i,:]) for t1 in -l1:l1, t2 in -l2:l2, dims=(1,2))
        # display((size(G[i,:,:]), size(sumkernal)))
        G[i,:,:] = G[i,:,:] ./ repeat(sumkernal, 1, n)'
        dAreshape[i,:,:] = dAreshape[i,:,:] ./ repeat(sumkernal, 1, n)'
    end
    gx, dgx = zeros(n,m), zeros(n,n,m)
    for j in 1:m
        gx[:,j] = G[:,:,j] * xmat[:,j]
        dgx[:,:,j] = G[:,:,j] + diagm(dAreshape[:,:,j] * xmat[:,j])
    end

    return gx, dgx
end



function posterior_XgY(μ_x, invΓ_x, invΓ_ϵ, G)
    Γ_xy = inv(G' * invΓ_ϵ * G + invΓ_x)
    # KKT = Γ_x * G' * inv(Γ_ϵ + G * Γ_x * G') * G * Γ_x
    # Γ_xy_woodbury = Γ_x - KKT
    μ_xy = Γ_xy * (invΓ_x * μ_x + G' * invΓ_ϵ * y)
    return μ_xy, Γ_xy
end

function posterior_XgY_lowRank(μ_x, Γ_x, invΓ_ϵ, G, r)
    invΓ_x = inv(Γ_x)
    L = cholesky(Γ_x * diagm(ones(length(μ_x)))).L
    trilEig = tril(L' * G' * invΓ_ϵ * G * L) + tril(L' * G' * invΓ_ϵ * G * L, -1)'# - diagm(diag(L' * G' * invΓ_ϵ * G * L))
    Λ_orig1, Q_orig1 = eigen(trilEig)

    Γ = Λ_orig1[end:-1:1]
    V = Q_orig1[:,end:-1:1]
    W = L * V
    # r = 200 # length(Γ)
    Γ_xy_tilde = Γ_x - W[:,1:r] * diagm(Γ[1:r]) * inv(diagm(Γ[1:r] + ones(r))) * W[:,1:r]'
    μ_xy_tilde = Γ_xy_tilde * (invΓ_x * μ_x +  G' * invΓ_ϵ * y)
    return μ_xy_tilde, Γ_xy_tilde, Γ
end

function posterior_ZgY(μ_z, invΓ_z, Γ_x, Γ_ϵ, G, O)
    Γ_Δ = Γ_ϵ + G * (Γ_x - Γ_x * O' * invΓ_z * O * Γ_x) * G'
    F = G * Γ_x * O' * invΓ_z
    invΓ_Δ = inv(Γ_Δ)
    Γ_zy = inv(F' * invΓ_Δ * F + invΓ_z)
    μ_zy = Γ_zy * (invΓ_z * μ_z + F' * invΓ_Δ * y)
    return μ_zy, Γ_zy
end

function posterior_ZgY_lowRank(μ_z, invΓ_z, Γ_x, Γ_ϵ, G, O, r)

    Γ_y = tril(Γ_ϵ + G * Γ_x * G') + tril(Γ_ϵ + G * Γ_x * G', -1)' #- diagm(diag(tril(Γ_ϵ + G * Γ_x * G')))
    invL = inv(cholesky(Γ_y).L)
    nonSym = invL * G * Γ_x * O' * invΓ_z * O * Γ_x * G' * invL'

    Λ_orig, Q_orig = eigen(tril(nonSym) + tril(nonSym)' - diagm(diag(nonSym)))
    Λ = Λ_orig[end:-1:1]
    Q = invL' * Q_orig[:,end:-1:1]
    Q_hat = O * Γ_x * G' * Q

    Γ_zy_tilde = Γ_z - Q_hat[:,1:r] * diagm(Λ[1:r]) * Q_hat[:,1:r]'
    μ_zy_tilde = Q_hat[:,1:r] * diagm(Λ[1:r]) * Q[:,1:r]' * y + Γ_zy_tilde * invΓ_z * μ_z
    return μ_zy_tilde, Γ_zy_tilde, Λ
end

function posterior_ZgY_lowRank_eig(invΓ_z, Γ_x, Γ_ϵ, G, O)

    Γ_y = tril(Γ_ϵ + G * Γ_x * G') + tril(Γ_ϵ + G * Γ_x * G', -1)' #- diagm(diag(tril(Γ_ϵ + G * Γ_x * G')))
    invL = inv(cholesky(Γ_y).L)
    nonSym = invL * G * Γ_x * O' * invΓ_z * O * Γ_x * G' * invL'

    Λ_orig, Q_orig = eigen(tril(nonSym) + tril(nonSym, -1)')
    Λ = Λ_orig[end:-1:1]
    Q = invL' * Q_orig[:,end:-1:1]
    Q_hat = O * Γ_x * G' * Q

   
    return Λ, Q, Q_hat
end