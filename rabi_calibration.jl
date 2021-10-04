using IonSim
using QuantumOptics: timeevolution, stochastic
using StochasticDiffEq
using ScikitLearn
using Random
using Optim
using Distributions

function rabi_calibration(detuning, pi_time, initial_guess, N, tspan_ideal, tspan_experiment, timescale=1e-6)
    ca_ion = Ca40(["S-1/2", "D-1/2"])
    chain = LinearChain(
        ions=[ca_ion],
        com_frequencies=(x=3e6,y=3e6,z=1e6), 
        vibrational_modes=(;z=[1]),
    )
    laser = Laser(
        k = (x̂ + ẑ)/√2,
        ϵ = (x̂ - ẑ)/√2,
    )
    trap = Trap(configuration=chain, B=4e-4, Bhat=ẑ, δB=0, lasers=[laser])
    laser.Δ = transition_frequency(trap, 1, ("S-1/2", "D-1/2"))

    function simulate_trap(tspan)
        detuned_laser = copy(laser)
        detuned_laser.Δ += detuning
        trap.lasers = [detuned_laser]
        Efield_from_pi_time!(pi_time*timescale, trap, 1, 1, ("S-1/2", "D-1/2"));    

        h = hamiltonian(trap, timescale=timescale)
        mode = trap.configuration.vibrational_modes.z[1]
        @time tout, sol = timeevolution.schroedinger_dynamic(tspan, ionstate(trap, "S-1/2") ⊗ mode[0], h)
        ex = real.(expect(ionprojector(trap, "D-1/2"), sol))
        return tout, ex
    end;

    tout, ideal_ex = simulate_trap(tspan_ideal);

    # function to simulate taking real data
    function sample(expectation_values)
        samples = []
        for p in expectation_values
            number = rand(Float64)
            if p <= number
                push!(samples, 0)
            else
                push!(samples, 1)
            end
        end
        return samples
    end;

    function simulate_experiment(n_shots)
        _, expectation_values = simulate_trap(tspan_experiment)
        counts = sample(expectation_values)
        for n = 2:n_shots
            counts += sample(expectation_values)
        end
        return counts
    end;


    # let's just use the traditional pure state theory
    function prob_1(time, θ₁, θ₂)
        sigX = [0 1; 1 0]
        sigZ = [1 0; 0 -1]
        H = 0.5*θ₁*sigX - 0.5*θ₂*sigZ
        U = exp(-im*H*time)
        return abs([0, 1]'*U*[1, 0])^2
    end;

    function log_likelihood(bright_counts, num_experiments, times, θ₁, θ₂)
        out = 0
        for i in eachindex(times)
            p_1 = prob_1(times[i], θ₁, θ₂)
            dist = Multinomial(num_experiments, [1-p_1, p_1])
            term = logpdf(dist, [num_experiments-bright_counts[i], bright_counts[i]])
            if term != NaN
                out += term
            end
        end
        return out
    end;

    # run the experiment, learn the model, and reconstruct the curve from the learned model
    bright_counts = simulate_experiment(N);
    function objective(θ::Vector)
        return -log_likelihood(bright_counts, N, tspan_experiment, θ[1], θ[2])
    end
    res = optimize(objective, initial_guess)
    θ₁, θ₂ = Optim.minimizer(res)
    predicted_ex = []
    for t in tspan_ideal
        push!(predicted_ex, prob_1(t, θ₁, θ₂))
    end
    
    # also find the fit params for the ideal curve
    ideal_bright_counts = simulate_experiment(1000000);
    function objective_ideal(θ::Vector)
        return -log_likelihood(ideal_bright_counts, 1000000, tspan_experiment, θ[1], θ[2])
    end
    res = optimize(objective_ideal, initial_guess)
    ideal_θ₁, ideal_θ₂ = Optim.minimizer(res)
    
    return Dict(
        "ideal_curve" => ideal_ex,
        "ideal_fit_params" => [ideal_θ₁, ideal_θ₂],
        "experimental_data" => bright_counts / N,
        "learned_curve" => predicted_ex,
        "learned_fit_params" => [θ₁, θ₂],
    )
end

function rabi_fidelity(actual_detuning, actual_pi_time, learned_detuning, learned_pi_time, timescale=1e-6)
    ca_ion = Ca40(["S-1/2", "D-1/2"])
    chain = LinearChain(
        ions=[ca_ion],
        com_frequencies=(x=3e6,y=3e6,z=1e6), 
        vibrational_modes=(;z=[1]),
    )
    laser = Laser(
        k = (x̂ + ẑ)/√2,
        ϵ = (x̂ - ẑ)/√2,
    )
    trap = Trap(configuration=chain, B=4e-4, Bhat=ẑ, δB=0, lasers=[laser])
    laser.Δ = transition_frequency(trap, 1, ("S-1/2", "D-1/2"))
    
    # adjust the laser frequency by the error in the learned detuning
    detuned_laser = copy(laser)
    learned_detuning_error = actual_detuning - learned_detuning
    detuned_laser.Δ += learned_detuning_error
    trap.lasers = [detuned_laser]
    
    # the E-field corresponds to the actual pi time, but we will evolve for the learned pi time
    Efield_from_pi_time!(actual_pi_time*timescale, trap, 1, 1, ("S-1/2", "D-1/2"));
    tspan = [0, learned_pi_time]

    h = hamiltonian(trap, timescale=timescale)
    mode = trap.configuration.vibrational_modes.z[1]
    @time tout, sol = timeevolution.schroedinger_dynamic(tspan, ionstate(trap, "S-1/2") ⊗ mode[0], h)
    ex = real.(expect(ionprojector(trap, "D-1/2"), sol))
    return ex[end]
end