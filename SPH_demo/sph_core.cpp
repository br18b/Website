#include <emscripten.h>
#include <math.h>
#include <vector>
#include <set>

extern "C" {

    struct Particle {
        float x, y;
        float vx, vy;
        float ax, ay;
        float rho, p;
        float T;
        int flag; // boundary, lid etc. Bitwise (!) flag
        int myCellIndex;
    };

    // Parameters
    const int nW = 10000;
    const int pressureTableSize = 100000;
    float pressureTable[pressureTableSize + 1];
    const float rhoMax = 100.0f;
    constexpr float T_TDM = 273.15;
    float h = 20.0f;
    float h20 = h * h;
    float restDensity = 0.5;
    float mass = 1.0f;
    float mu = 10.0f;
    float mu_target = mu;
    float g = -0.1f;
    float k = 100.0f;
    float T0 = 50.0f;
    float alpha = 0.1f;
    float T_diff = 0.01;

    float mu_relax = 100;

    float dt_prev = 1;

    float cacc = 0.3;
    float cfl = 0.25;

    float amax = 0;
    float vmax = 0;

    float aRho = restDensity / (1 - alpha * (T_TDM + T0));
    float bRho = alpha / (1 - alpha * (T_TDM + T0));

    float xmin = 100;
    float xmax = 800;
    float ymin = 100;
    float ymax = 800;

    std::vector<float> h2, rho0;
    bool h2_initialized = false;

    EMSCRIPTEN_KEEPALIVE
    void set_simulation_params(float rho0_, float h_, float mass_, float mu_, float g_, float k_, float T0_, float T_diff_, float alpha_, float xmin_, float xmax_, float ymin_, float ymax_) {
        restDensity = rho0_,
        h = h_;
        h20 = h * h;
        mass = mass_;
        mu_target = mu_;
        mu = mu_target;
        g = g_;
        k = k_;
        T0 = T0_;
        T_diff = T_diff_;
        alpha = alpha_;
        xmin = xmin_;
        xmax = xmax_;
        ymin = ymin_;
        ymax = ymax_;
        aRho = restDensity / (1 - alpha * (T_TDM + T0));
        bRho = alpha / (1 - alpha * (T_TDM + T0));
    
        float drho = rhoMax / pressureTableSize;
        for (int i = 0; i < pressureTableSize + 1; i++) {
            float rho = i * drho;
            if (rho <= 0.0f)
                pressureTable[i] = -1.0f;  // or clamp/log warning etc.
            else
                pressureTable[i] = powf(rho, k) - 1.0f;
        }
    }

    static_assert(sizeof(Particle) == 11 * 4, "Particle size mismatch");

    // Grid
    int gridW = 0, gridH = 0;
    float originX = 0.0f, originY = 0.0f, cellSize = h;
    std::vector<std::vector<int>> grid;

    float W_lookup[nW + 1];
    float gradW_lookup[nW + 1];

    inline int gridIndex(int ix, int iy) {
        return iy * gridW + ix;
    }

    // Add globals to store bounds
    float simXmin, simXmax, simYmin, simYmax;

    EMSCRIPTEN_KEEPALIVE
    void setup_grid(float x0, float y0, float x1, float y1, float cs) {
        originX = x0;
        originY = y0;
        simXmin = x0;
        simXmax = x1;
        simYmin = y0;
        simYmax = y1;
        cellSize = cs;

        gridW = (int)ceil((x1 - x0) / cs);
        gridH = (int)ceil((y1 - y0) / cs);
        grid.assign(gridW * gridH, {});
    }

    EMSCRIPTEN_KEEPALIVE
    void fill_grid(Particle* particles, int N) {
        //printf("📌 Entered fill_grid with %d particles\n", N); fflush(stdout);
        for (auto& cell: grid) cell.clear(); // clear old entries
        for (int i = 0; i < N; i++) {
            Particle& p = particles[i];
            int ix = (int)((p.x - originX) / cellSize);
            int iy = (int)((p.y - originY) / cellSize);
            if (ix >= 0 && ix < gridW && iy >= 0 && iy < gridH) {
                int cellIndex_unrolled = gridIndex(ix, iy);
                grid[cellIndex_unrolled].push_back(i);
                particles[i].myCellIndex = grid[cellIndex_unrolled].size() - 1;
            }
        }
        for (auto& cell: grid) {
            int estimated_max_size = cell.size();
            estimated_max_size = 2 * std::max(estimated_max_size, N / (gridW * gridH));
            cell.reserve(estimated_max_size);
        }
    }

    EMSCRIPTEN_KEEPALIVE
    void init_W_lookup() {
        //printf("📌 Entered init_W_lookup\n"); fflush(stdout);
        for (int i = 0; i <= nW; i++) {
            //float r2 = (float)i / nW * h2;
            //float q = sqrtf(r2 / h2);
            float q = sqrt((float)i / nW);
            if (i == nW) {
                W_lookup[i] = 0;
                gradW_lookup[i] = 0;
            }
            else {
                W_lookup[i] = (1 - q) * (1 - q) * (1 - q); // 3.183098861837907 = 10 / pi. Factor of 1/h^2 is still missing
                if (i == 0) {
                    gradW_lookup[i] = 0;
                }
                else {
                    gradW_lookup[i] = - 3 * (1 - q) * (1 - q) / q; // this is 9.549296585513721 = 30 / pi. leftover 1/h^4 for normalization must be multiplied inline
                }
            }
            //W_lookup[i] = (q >= 1.0f) ? 0.0f : powf(1.0f - q, 3.0f);
            //gradW_lookup[i] = (q >= 1.0f || q == 0.0f) ? 0.0f : -3.0f * powf(1.0f - q, 2.0f) / (h * sqrtf(r2));
        }
    }

    EMSCRIPTEN_KEEPALIVE
    void compute_density_forces(Particle* particles, int N) {
        // Step 1: compute density for all particles
        //printf("📌 Started density loop with %d particles\n", N); fflush(stdout);
        // precompute kernel sizes
        if (!h2_initialized) {
            h2.resize(N);
            rho0.resize(N);
            h2_initialized = true;
        }
        for (int i = 0; i < N; i++) {
            float T = particles[i].T;
            rho0[i] = restDensity * ((T_TDM + T0) / (T_TDM + T)) * ((T_TDM + T0) / (T_TDM + T));
            h2[i] = h20 * restDensity / rho0[i];
        }
        for (int i = 0; i < N; i++) {
            Particle& pi = particles[i];
            pi.rho = 0.0f;

            int pix = (int)((pi.x - originX) / cellSize);
            int piy = (int)((pi.y - originY) / cellSize);
            
            float this_h2 = h2[i];
            for (int dx = -1; dx <= 1; dx++) {
                for (int dy = -1; dy <= 1; dy++) {
                    int nix = pix + dx;
                    int niy = piy + dy;
                    if (nix < 0 || nix >= gridW || niy < 0 || niy >= gridH) continue;
    
                    for (int j : grid[gridIndex(nix, niy)]) {
                        /*if (j < 0 || j >= N) {
                            printf("Invalid particle index j=%d in cell %d\n", j, gridIndex(nix, niy));
                            fflush(stdout);
                            abort();
                        }*/
                        float dx = pi.x - particles[j].x;
                        float dy = pi.y - particles[j].y;
                        float r2 = dx * dx + dy * dy;
                        float h2_avg = 0.5 * (this_h2 + h2[j]);
                        if (r2 >= h2_avg) continue;
    
                        int idx = (int)(r2 * nW / h2_avg);
                        /*if (idx < 0 || idx > nW) {
                            printf("Bad kernel index: %d (r2 = %f) while computing density", idx, r2);
                            fflush(stdout);
                            abort();
                        }*/
                        //if (idx > nW) idx = nW;
                        float W = W_lookup[idx];// / h2_avg;
                        pi.rho += mass * W;
                    }
                }
            }
            //float rho0 = restDensity * (1.0f - alpha * (pi.T - T0));
            //pi.p = k * (pi.rho - rho0);
            //pi.p = std::powf(pi.rho / rho0, k) - 1;
            /*if (!std::isfinite(pi.rho) || pi.rho < 0.0f) {
                printf("Bad rho for particle %d: rho = %f\n", i, pi.rho);
                fflush(stdout);
            */
            int idRho = std::min((int)((pi.rho / rho0[i]) * (pressureTableSize / rhoMax)), pressureTableSize);
            //printf("rho0: %f, T: %f, idx: %d\n", rho0, pi.T, idRho); fflush(stdout);
            pi.p = pressureTable[idRho];
        }
    
        // Step 2: compute pressure + forces
        //printf("📌 Started force loop with %d particles\n", N); fflush(stdout);
        amax = 0; vmax = 0;
        for (int i = 0; i < N; i++) {
            Particle& pi = particles[i];
            if (pi.flag & 1) continue;

            float ax = 0.0f, ay = 0.0f;

            float T_laplacian = 0.0f;
            float weightSum = 0.0f;
    
            int pix = (int)((pi.x - originX) / cellSize);
            int piy = (int)((pi.y - originY) / cellSize);
            
            float this_h2 = h2[i];
            for (int dx = -1; dx <= 1; dx++) {
                for (int dy = -1; dy <= 1; dy++) {
                    int nix = pix + dx;
                    int niy = piy + dy;
                    if (nix < 0 || nix >= gridW || niy < 0 || niy >= gridH) continue;
    
                    for (int j : grid[gridIndex(nix, niy)]) {
                        /*if (j < 0 || j >= N) {
                            printf("Invalid particle index j=%d in cell %d\n", j, gridIndex(nix, niy));
                            fflush(stdout);
                            abort();
                        }*/
                        if (i == j) continue;
                        Particle& pj = particles[j];
    
                        float dx = pi.x - pj.x;
                        float dy = pi.y - pj.y;
                        float r2 = dx * dx + dy * dy;
                        float h2_avg = 0.5 * (this_h2 + h2[j]);

                        if (r2 >= h2_avg) continue;
    
                        int idx = (int)(r2 * nW / h2_avg);
                        /*if (idx < 0 || idx > nW) {
                            printf("Bad kernel index: %d (r2 = %f) while computing forces", idx, r2);
                            fflush(stdout);
                            abort();
                        }*/
                        //if (idx > nW) idx = nW;
                        float W = W_lookup[idx];// / h2_avg;
                        float dW = gradW_lookup[idx];// / (h2_avg * h2_avg);
    
                        float gradx = dW * dx;
                        float grady = dW * dy;
    
                        //float f = -mass * (pi.p + pj.p) / (2.0f * pj.rho);
                        float f = -mass * (pi.p / (pi.rho * pi.rho) + pj.p / (pj.rho * pj.rho));
                        ax += f * gradx;
                        ay += f * grady;
    
                        float dvx = pj.vx - pi.vx;
                        float dvy = pj.vy - pi.vy;
                        float mu_corr = 1 + 0.5 * (pi.T + pj.T) / 273.15;
                        float mu_term = (mu / mu_corr) * mass * (dvx * gradx + dvy * grady) / (pi.rho + pj.rho);
                        ax += mu_term * gradx;
                        ay += mu_term * grady;

                        T_laplacian += W * (pj.T - pi.T);
                        weightSum += W;
                    }
                }
            }

            if (weightSum > 0.0f) {
                pi.T += (restDensity / pi.rho) * (1 + pi.T / T_TDM) * (1 + pi.T / T_TDM) * T_diff * T_laplacian / weightSum; // more diffusion for hot regions
            }
    
            //ay += -pi.rho * g / mass;
            ay += (pi.T - T0) * g;
            pi.ax = ax;
            pi.ay = ay;

            amax = std::max(pi.ax * pi.ax + pi.ay * pi.ay, amax);
            vmax = std::max(pi.vx * pi.vx + pi.vy * pi.vy, vmax);
        }
        amax = std::sqrt(amax);
        vmax = std::sqrt(vmax);
    }
    
    EMSCRIPTEN_KEEPALIVE
    void integrate(Particle* particles, int N, float* dt_max_ptr) {
        float dt = *dt_max_ptr;
        if (vmax > 0) {
            dt = std::min(dt, cfl * h / vmax);
        }
        if (amax > 0) {
            dt = std::min(dt, cacc * std::sqrt(h / amax));
        }
        if (dt > dt_prev) dt = 0.5 * dt + 0.5 * dt_prev;
        dt_prev = dt;
        *dt_max_ptr = dt;
        mu -= (mu - mu_target) * dt * mu_relax;
        //printf("current mu: %f\n", mu);
        //fflush(stdout);
        //printf("dt: %f, vmax: %f, amax: %f\n", dt, vmax, amax);
        //fflush(stdout);
        for (int i = 0; i < N; i++) {
            Particle& p = particles[i];
            if (p.flag & 1) continue;
            
            // store the old indices in case p changes cells
            int pix = (int)((p.x - originX) / cellSize);
            int piy = (int)((p.y - originY) / cellSize);

            float x_old = p.x;
            float y_old = p.y;

            // Integrate
            p.vx += p.ax * dt;
            p.vy += p.ay * dt;
            p.x += p.vx * dt;
            p.y += p.vy * dt;

            /*if (!std::isfinite(p.x) || !std::isfinite(p.y)) {
                printf("Particle %d exploded: x=%f y=%f\n", i, p.x, p.y);
                fflush(stdout);
                p.x = originX; p.y = originY;
                p.vx = p.vy = p.ax = p.ay = 0;
            }*/

            // out of bounds?
            if (p.x < xmin) {
                p.vx = -p.vx;
                p.x = 2 * xmin - p.x;
            }
            if (p.x > xmax) {
                p.vx = -p.vx;
                p.x = 2 * xmax - p.x;
            }
            if (p.y < ymin) {
                p.vy = -p.vy;
                p.y = 2 * ymin - p.y;
            }
            if (p.y > ymax) {
                p.vy = -p.vy;
                p.y = 2 * ymax - p.y;
            }

            // Reset acceleration
            p.ax = 0.0f;
            p.ay = 0.0f;

            // New grid cell
            int new_ix = (int)((p.x - originX) / cellSize);
            int new_iy = (int)((p.y - originY) / cellSize);

            if (new_ix != pix || new_iy != piy) {
                /*if (pix < 0 || pix >= gridW || piy < 0 || piy >= gridH) {
                    printf("Particle %d went out of bounds: x = %f y = %f, ix = %d, iy = %d\n", i, p.x, p.y, pix, piy);
                    fflush(stdout);
                    abort();
                }
                if (new_ix < 0 || new_ix >= gridW || new_iy < 0 || new_iy >= gridH) {
                    printf("Particle %d updated position went out of bounds: x = %f y = %f, ix = %d, iy = %d\n", i, p.x, p.y, new_ix, new_iy);
                    printf("Old position: x = %f y = %f, xmin: %f, xmax: %f, ymin: %f, ymax: %f\n", x_old, y_old, xmin, xmax, ymin, ymax);
                    fflush(stdout);
                    abort();
                }*/

                // to get rid of the particle from the old grid cell we perform the 'switcheroo - pop' O(1) trick
                // this trick allows us to effectively erase an element from the middle of a vector, without moving a bunch of elements past the erasure position
                // move - move the back element to the position of the undersired particle and pop back
                int oldIndex = gridIndex(pix, piy); // index of the cell from which we want to remove the particle
                int oldPos = p.myCellIndex; // index position of the particle within the old cell

                int lastIdx = grid[oldIndex].back(); // index position of the last particle within the old cell
                grid[oldIndex].pop_back(); // we get rid of this element, but restore it at the oldPos later

                if (oldPos != grid[oldIndex].size()) { // if, accidentally, the erased particle was at the back of the vector, we don't have to do anything else...
                    grid[oldIndex][oldPos] = lastIdx; // this index position gets replaced with the back
                    particles[lastIdx].myCellIndex = oldPos; // back particle's index position within the cell was affected by the switcheroo, so we update it, too
                }

                // now we add the particle to the shiny new cell by just pushing it back
                int newIndex = gridIndex(new_ix, new_iy);
                grid[newIndex].push_back(i);
                p.myCellIndex = grid[newIndex].size() - 1;

                // Update stored cell indices
                //p.ix = new_ix;
                //p.iy = new_iy;
            }
        }
    }
} // extern "C"