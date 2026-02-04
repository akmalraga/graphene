# 3D Kane-Mele Model: Anderson Disorder & Topology

Repositori ini berisi kumpulan skrip simulasi numerik menggunakan **PythTB** untuk mengeksplorasi fase topologis pada model Kane-Mele 3D. Penelitian ini berfokus pada transisi fase antara isolator topologis dan isolator trivial akibat pengaruh *staggered potential*, *Rashba coupling*, dan gangguan atomik (*Anderson disorder*).

## 🔬 Hamiltonian Model
Model ini dibangun menggunakan pendekatan *tight-binding* pada kisi hexagonal bertumpuk (stacked honeycomb lattice). Hamiltonian yang digunakan mencakup empat suku utama:

$$
H = -t \sum_{\langle i,j \rangle} c_i^\dagger c_j + i \lambda_{SO} \sum_{\langle \langle i,j \rangle \rangle} \nu_{ij} c_i^\dagger s_z c_j + \lambda_R H_R + \Delta \sum_{i} \xi_i c_i^\dagger c_i
$$

1. **$t$**: Lonjakan elektron antar tetangga terdekat (*Nearest Neighbor*).
2. **$\lambda_{SO}$**: Interaksi spin-orbit intrinsik (membuka celah topologis).
3. **$\lambda_R$**: Interaksi Rashba (merusak simetri inversi).
4. **$\Delta$**: Potensial subkisi (mendorong sistem ke fase trivial).
5. **Anderson Disorder**: Fluktuasi onsite acak yang disimulasikan melalui distribusi uniform.

---

## 📂 Daftar Skrip dan Kegunaan

| File | Deskripsi |
| :--- | :--- |
| `hwc_my.py` | Menghitung **Hybrid Wannier Centers** (Wilson Loop) untuk memverifikasi topologi bulk. |
| `fase.py` | Membuat galeri spektrum ribbon untuk melihat transisi fase secara visual. |
| `disorder_av_new.py` | Menghitung rata-rata spektrum energi (ensemble average) dengan kehadiran **Anderson Disorder**. |
| `vis_mymodel.py` | Visualisasi struktur pita energi (*band structure*) bulk 3D. |
| `slab_mymodel.py` | Simulasi sistem slab untuk mengamati *surface states*. |
| `data2dos2.py` | Ekstraksi data **Density of States (DOS)** pada tingkat energi Fermi ($E=0$). |

---

## 📊 Hasil Simulasi
Simulasi ini bertujuan untuk menunjukkan bahwa keadaan tepi (*edge states*) tetap kokoh (robust) selama celah energi topologis belum tertutup oleh potensial $\Delta$ atau gangguan disorder $W$.



---

## 🚀 Cara Menjalankan
1. **Prasyarat**:
   Pastikan Anda telah menginstal pustaka yang diperlukan:
   ```bash
   pip install numpy matplotlib pythtb pandas
