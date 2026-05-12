# 1-D Quantum: FEDVR DVR basis × Jordan-Wigner

`Lab/BiqBird/Sample/1D/FEDVR/` で完成した古典 RHF / FCI 級 1D ソフトクーロン量子化学を、
**FEDVR の DVR ノードを直接 qubit にエンコード**し、
**Jordan-Wigner（JW）変換**で fermion から spin (qubit) へ落とし込み、
理想量子コンピュータでの基底状態探索（QPE）と時間発展（Trotter）を
古典シミュレータ上で実装する。

## 目次

0. [古典側と量子側の役割分担](#0-古典側と量子側の役割分担)
1. [物理ハミルトニアンと DVR 第二量子化形式](#1-物理ハミルトニアンと-dvr-第二量子化形式)
2. [Spin-orbital のフェルミオンモードへの埋め込み](#2-spin-orbital-のフェルミオンモードへの埋め込み)
3. [Jordan-Wigner 変換と qubit Hamiltonian](#3-jordan-wigner-変換と-qubit-hamiltonian)
4. [粒子数セクター直接構築（メモリ最適化）](#4-粒子数セクター直接構築メモリ最適化)
5. [Solver A : 古典直接対角化（reference）](#5-solver-a--古典直接対角化reference)
6. [Solver B : Iterative QPE（量子）](#6-solver-b--iterative-qpe量子)
7. [Solver C : Trotter 時間発展（量子）](#7-solver-c--trotter-時間発展量子)
8. [データフローと JSON スキーマ](#8-データフローと-json-スキーマ)
9. [段階的検証結果（Phase Q1〜Q4）](#9-段階的検証結果phase-q1q4)
10. [既存 FD/FEDVR・連続極限値との比較](#10-既存-fdfedvr連続極限値との比較)
11. [ビルド・実行手順](#11-ビルド実行手順)
12. [Qiskit による量子回路レベルでの再構成（Phase Q5）](#12-qiskit-による量子回路レベルでの再構成phase-q5)
13. [参考文献](#13-参考文献)

---

## 0. 古典側と量子側の役割分担

本実装の各レイヤーが古典コンピュータ上で実行する処理か、量子コンピュータ上で実行することが想定される処理かを以下に整理する。Phase Q1〜Q5 のすべての結果は古典シミュレータ上で得られたものであり、§0.2 で述べる理由から量子計算が同じ Hamiltonian に対して返す値の予測値となる。

### 0.1 レイヤー別の実行場所

| レイヤー | 処理内容 | 実装 | 種別 |
|---|---|---|---|
| ① 積分構築 | $h_{pq}, V_{pq}$ の計算 | C++（`fedvr_basis.hpp`, `export_integrals.cpp`）| 古典前処理 |
| ② Hamiltonian 構築 | FermionOperator → JW → QubitOperator | Python（`build_qubit_h.py`, OpenFermion）| 古典前処理 |
| ③ 状態の表現 | 状態ベクトル（ $2^n$ 次元複素配列） | numpy 配列 / Qiskit Statevector | 古典シミュレータ（量子回路で扱う対象） |
| ④ 直接対角化 (Solver A) | $H$ の最低固有値計算 | Python（`direct_diag.py`、scipy）| 古典 reference |
| ⑤ iQPE (Solver B) | controlled- $U^{2^k}$ + ancilla 測定 | Python（`iqpe.py`, `q5_qiskit_reference.py`）| 古典シミュレータが量子回路を再現 |
| ⑥ Trotter (Solver C) | $U(\Delta t)$ を分割して時間発展 | Python（`trotter.py`）| 古典シミュレータが量子回路を再現 |

古典前処理 (①②) は実機上で量子計算を行う場合でも事前計算の段階で必要な処理である。古典 reference (④) は量子計算結果と比較する基準値を提供する。古典シミュレータ (③⑤⑥) はユニタリ変換と射影測定を数値的に展開し、本来量子コンピュータが実行する処理を古典で再現する。

### 0.2 古典シミュレータが量子計算の予測値を与える根拠

理想量子コンピュータの動作はユニタリ変換と射影測定の組み合わせで完全に記述される。古典シミュレータ（numpy + scipy + Qiskit Aer）でユニタリ行列を状態ベクトルに作用させ、必要に応じて測定統計を乱数で再現すれば、得られる確率分布と期待値は実機が同じ回路を実行したときの値と一致する。

ただし古典シミュレータ上では $N_q$ qubit 状態の保持に $2^{N_q}$ サイズの複素配列を要するため、qubit 数が増えると指数的に資源が必要になる。本実装の対象範囲（ $N_q \le 28$ ）では古典側で完結するが、より大きな系での量子計算実行は古典シミュレータでは扱えない。

### 0.3 古典シミュレータ実装で前提として用いている事項

実装上、以下の 2 点については古典側で得られる情報を利用している。これらは古典シミュレータでの計算量を抑えるための実装選択であり、本実装で得た数値結果と量子実機が同じ Hamiltonian に対して返す値が一致することを保証する範囲内で行っている。

#### (A) iQPE における固有状態前提（[`iqpe.run_iqpe`](biqbird_quantum/iqpe.py)）

iQPE プロトコルではラウンド $k$ で controlled- $U^{2^{m}}$ （ $m = n_{\mathrm{bits}} - 1 - k$ ）を ancilla 制御として作用させる必要がある。古典シミュレータでこれをそのまま実行すると `scipy.sparse.linalg.expm_multiply(-i τ · 2^{n-1-k} H, ψ)` を呼ぶことになり、 $\tau \cdot 2^{n-1} \cdot \|H\|$ が大きいときに Krylov 部分空間の degree が増大して計算時間が現実的でなくなる。

`run_iqpe` では入力 $|\psi\rangle$ が $H$ の固有状態（直接対角化で得たもの）であることを利用し、固有関係

$$
U^{2^k} |\psi\rangle = e^{-i E \cdot 2^k \cdot \tau} |\psi\rangle
$$

から ancilla の $|1\rangle$ 分岐に付く位相を解析的に決定する。この場合行列ベクトル積は実行されず、計算時間は ancilla 操作のみに支配される。

固有状態前提が成立しない一般的な入力に対しては `run_iqpe_general` で `expm_multiply` を経由する実装も提供している。Phase Q5（§12）では、Qiskit 上で実際に controlled- $U^{2^k}$ を量子回路として組み、shots ベースのサンプリング測定を経由した結果が `run_iqpe` の解析計算と一致することを確認している。

#### (B) Trotter におけるセクター射影（[`trotter.evolve`](biqbird_quantum/trotter.py)）

$\hat H_{\mathrm{el}}$ は粒子数演算子 $\hat N$ と可換であり、 $N_e$ 電子セクターは閉じた部分空間をなす。`trotter.evolve` はこの部分空間上で時間発展を計算するため、§4 で述べた粒子数射影 $H_{\mathrm{sec}} \in \mathbb{C}^{D \times D}$ （ $D = \binom{N_q}{N_e}$ ）を入力として受け取る。He 28 qubit でも $D = 378$ となり、古典シミュレータのメモリ制約内に収まる。

実機上で同じ Trotter を実行する場合、回路は計算基底全体（ $2^{N_q}$ 次元）に作用するが、各ゲートが $\hat N$ と可換になるよう設計すれば（DVR 縮約後の $\hat T$ と $\hat V$ はいずれも粒子数を保存する）、初期状態の粒子数セクターに留まる。古典シミュレータで陽にセクター射影を行うか、計算基底全体で動かして対称性により自動的にセクターに留まるかという違いであり、得られる物理量は一致する。

### 0.4 量子実機での実行を想定する場合のレイヤー構成

量子実機上での実行を想定する場合、§1〜§7 で構築した Hamiltonian はそのまま古典前処理として再利用し、以下を量子回路として実行する。

1. **古典前処理**: ①② を従来通り行い、JW 変換後の `QubitOperator`（または Qiskit `SparsePauliOp`）を得る
2. **状態準備**: HF determinant に対応する bit string（例： $|110000\ldots\rangle$ ）を計算基底状態として初期化、もしくは adiabatic state preparation で基底状態に近づける
3. **iQPE 量子回路**: ancilla を $|+\rangle$ に置き、controlled- $U^{2^k}$ （内部は Trotter 分解）、 $R_z(-\omega_k)$ 、Hadamard、測定を $n_{\mathrm{bits}}$ 回繰り返す
4. **Trotter 回路**: iQPE の中で controlled- $U^{2^k}$ の本体として、 $e^{-i V \Delta t}$ を $R_z, R_{zz}$ ゲート列、 $e^{-i T \Delta t}$ を Givens 回転 / fSim ゲート列に展開する

このレイヤー構成における必要量子リソースの見積もりは §12.3 にまとめる。

---

## 1. 物理ハミルトニアンと DVR 第二量子化形式

### 1.1 規約

すべて原子単位、ソフトクーロンパラメータ $a = 1$ で固定（FD/FEDVR と共通）。
$N_n$ 個の核（電荷 $Z_a$ 、位置 $X_a$ ）下の $N_e$ 電子系：

$$
\hat H = \sum_{i=1}^{N_e}\left[-\frac{1}{2} \frac{d^2}{dx_i^2} - \sum_{a=1}^{N_n}\frac{Z_a}{\sqrt{(x_i - X_a)^2 + 1}}\right] + \sum_{i\lt j}\frac{1}{\sqrt{(x_i - x_j)^2 + 1}} + E_{\mathrm{nn}}
$$

$E_{\mathrm{nn}} = \sum_{a\lt b} Z_a Z_b / \sqrt{(X_a - X_b)^2 + 1}$ は核間反発。
量子計算では $E_{\mathrm{nn}}$ は外部から足すスカラー、電子部分のみを qubit に載せる。

### 1.2 DVR 基底（FEDVR から流用）

[`../FEDVR/fedvr_basis.hpp`](../FEDVR/fedvr_basis.hpp) の `FEDVRGrid(L, n_elements, n_order)` が返す
**直交化済み DVR 基底** $\lbrace\phi_p(x)\rbrace_{p=1,\dots,N_{\mathrm{DVR}}}$ を空間軌道とする
（ $N_{\mathrm{DVR}} = N_e^{\mathrm{elem}} \cdot n_{\mathrm{order}} - 1$ ）。
DVR の定義性質は

$$
\phi_p(x_q) = \delta_{pq}/\sqrt{w_p}, \quad \langle \phi_p | \phi_q \rangle \approx \delta_{pq} \quad (\text{GLL 求積})
$$

で、ノード座標 $\lbrace x_p \rbrace$ と DVR 重み $\lbrace w_p \rbrace$ がペアで揃う。
FEDVR ブリッジ点の規格化など詳細は [`../FEDVR/README.md`](../FEDVR/README.md) §2 を参照。

### 1.3 一電子・二電子積分

DVR 基底における **1 電子積分** $h_{pq}$ は

$$
h_{pq} = T_{pq} - \sum_a \frac{Z_a}{\sqrt{(x_p - X_a)^2 + 1}} \delta_{pq}
$$

（運動エネルギー $T$ は FEDVR で疎・バンド幅 $\sim 2 n_{\mathrm{order}} + 1$；
核引力は対角）。これは `fedvr::build_h_pq(grid, nuclei)` で構築済み。

**2 電子積分** は DVR の局所性により 4 添字 $V_{pqrs}$ が 2 添字に **縮約**：

$$
V_{pqrs} = \iint \phi_p(x_1)\phi_q(x_2) \frac{1}{\sqrt{(x_1-x_2)^2+1}} \phi_r(x_1)\phi_s(x_2) \ dx_1 dx_2 \approx \delta_{pr}\delta_{qs} V_{pq}
$$

ここで

$$
V_{pq} = \frac{1}{\sqrt{(x_p - x_q)^2 + 1}}, \quad V_{pp} = 1
$$

これが `fedvr::build_V_pq(grid)` の中身（ $N_{\mathrm{DVR}}^2$ サイズの密行列で十分）。
**Gaussian 基底に比べて Pauli string 数が劇的に少ない** 主要因がこの縮約である。

### 1.4 第二量子化 Hamiltonian

スピン軌道 $|\phi_p \otimes \sigma\rangle$ （ $\sigma \in \lbrace \alpha, \beta \rbrace$ ）に
生成消滅演算子 $c^\dagger_{p\sigma}, c_{p\sigma}$ を割り当てると、
DVR 縮約を代入した電子部分は

$$
\hat H_{\mathrm{el}} = \sum_{p,q,\sigma} h_{pq} \ c^\dagger_{p\sigma} c_{q\sigma} + \frac{1}{2} \sum_{p,q}\sum_{\sigma,\sigma'} V_{pq} \ c^\dagger_{p\sigma} c^\dagger_{q\sigma'} c_{q\sigma'} c_{p\sigma}
$$

$p = q$ かつ $\sigma = \sigma'$ の項は Pauli 排他律により $c^\dagger c^\dagger = 0$ で自動消滅。
全エネルギーは $E_{\mathrm{tot}} = E_{\mathrm{el}} + E_{\mathrm{nn}}$ 。

### 1.5 DVR の御利益（量子計算側で重要な点）

| | Gaussian basis | DVR basis (FEDVR) |
|---|---|---|
| 2電子積分テンソル | 4 添字 $V_{pqrs},\ O(N^4)$ 個 | 2 添字 $V_{pq},\ O(N^2)$ 個 |
| 第二量子化 H の項数 | $O(N^4) + O(N^2)$ | $O(N^2)$ + $O(N^2)$ |
| JW 後の Pauli string 数 | $O(N^4)$ | $O(N^2)$ |
| Trotter ゲート数 | $O(N^4)$ | $O(N^2)$ |

---

## 2. Spin-orbital のフェルミオンモードへの埋め込み

### 2.1 モードインデックス規約

OpenFermion の分子規約に揃え、空間軌道 $p \in \lbrace 0,\dots,N_{\mathrm{DVR}}-1 \rbrace$ と
スピン $s \in \lbrace 0(\alpha), 1(\beta) \rbrace$ を、フラットなフェルミオンモード番号 $k$ に：

$$
k = 2p + s \quad (0 \le k \lt  2 N_{\mathrm{DVR}})
$$

実装は `biqbird_quantum.build_qubit_h.spin_orbital_index(p, s)` で 1 行。

### 2.2 qubit register の物理的意味

JW 変換後、各フェルミオンモード $k$ は **そのまま qubit $k$** にマップされる：

$$
|0\rangle_k \equiv |\text{vacuum on mode } k\rangle, \quad |1\rangle_k \equiv c^\dagger_k |0\rangle_k
$$

つまり計算基底状態 $|x\rangle = |x_{N_q-1} \cdots x_1 x_0\rangle$ は
**Slater 行列式（占有数表現）と 1 対 1 対応**：

$$
|x\rangle = \prod_{k \ : \ x_k = 1} c^\dagger_k |\text{vac}\rangle
$$

総 qubit 数：

$$
N_q = 2 N_{\mathrm{DVR}} = 2 (N_e^{\mathrm{elem}} \cdot n_{\mathrm{order}} - 1)
$$

### 2.3 粒子数演算子と保存則

総粒子数演算子は

$$
\hat N = \sum_k c^\dagger_k c_k = \sum_k \frac{1 - Z_k}{2} = \frac{N_q}{2} - \frac{1}{2} \sum_k Z_k
$$

計算基底状態に作用すると **bit-popcount**： $\hat N |x\rangle = \mathrm{popcount}(x) |x\rangle$ 。
電子数 $N_e$ セクターは $\binom{N_q}{N_e}$ 次元で、 $\hat H_{\mathrm{el}}$ は粒子数を保存する
$([\hat H_{\mathrm{el}}, \hat N] = 0)$ ので、各セクターでブロック対角。
これがメモリ最適化（§4）の基礎。

---

## 3. Jordan-Wigner 変換と qubit Hamiltonian

### 3.1 JW 変換の定義

フェルミオン演算子の qubit 表現：

$$
c_k = \tfrac{1}{2} (X_k + i Y_k) \prod_{j\lt k} Z_j, \quad c^\dagger_k = \tfrac{1}{2} (X_k - i Y_k) \prod_{j\lt k} Z_j
$$

「 $j\lt k$ の Z 文字列」（**JW string**）が反交換子関係 $\lbrace c_j, c_k^\dagger \rbrace = \delta_{jk}$ を保証する。
等価表現：

$$
c_k = \prod_{j\lt k} Z_j \cdot \sigma^-_k, \quad \sigma^-_k = \tfrac{1}{2} (X_k + i Y_k)
$$

### 3.2 1-body 項のJW像

$h_{pq} c^\dagger_{p\sigma} c_{q\sigma}$ を JW で書き下す。
モード番号 $k_1 = 2p+s$ 、 $k_2 = 2q+s$ 、 $k_{\min} = \min(k_1, k_2)$ 、 $k_{\max} = \max(k_1, k_2)$ として、

- **$p = q$ （対角）**： $c^\dagger_k c_k = \frac{1 - Z_k}{2}$ → 1 個の $Z$ + 定数
- **$p \neq q$ （オフ対角、Hermite 化済み）**：

$$
c^\dagger_{k_1} c_{k_2} + \mathrm{h.c.} = \tfrac{1}{2} (X_{k_1} X_{k_2} + Y_{k_1} Y_{k_2}) \prod_{k_{\min} \lt  j \lt  k_{\max}} Z_j
$$

JW string は **モード番号で隣接していなくても** 間に挟まる Z たちで非局所性を持つが、
$p, q$ が DVR で隣接ノード同士なら $|k_1 - k_2| \le 2$ 程度に抑えられ、JW string も短い。

### 3.3 2-body 項のJW像

$V_{pq} c^\dagger_{p\sigma} c^\dagger_{q\sigma'} c_{q\sigma'} c_{p\sigma}$ は
**$p, q$ における占有数の積**で書き直せる：

$$
c^\dagger_p c^\dagger_q c_q c_p = \hat n_p \hat n_q = \frac{1-Z_p}{2}\cdot\frac{1-Z_q}{2} = \tfrac{1}{4} (I - Z_p - Z_q + Z_p Z_q)
$$

（ $p \neq q$ または $\sigma \neq \sigma'$ のとき；DVR 縮約のおかげでこれが成立）。
スピン和を取ると DVR 2-body 項は

$$
\tfrac{1}{2} V_{pq}\sum_{\sigma\sigma'} \hat n_{p\sigma} \hat n_{q\sigma'} = \tfrac{1}{2} V_{pq} (\hat n_{p\alpha} + \hat n_{p\beta})(\hat n_{q\alpha} + \hat n_{q\beta})
$$

すなわち **電荷密度演算子の積**。JW では各 $\hat n_k = (1 - Z_k)/2$ で、
**JW string が完全に消える**（ $Z$ の積 $\cdot Z = I$ で打ち消す）。
よって 2-body 項は **$Z_k$ と $Z_k Z_l$ の総和** だけで書ける（局所性が高い）。

これが「DVR + JW」の決定的な御利益：Gaussian + JW なら 2 添字を残した
8 項のテンソル分解が必要だが、DVR では **対角と $ZZ$ だけ**。

### 3.4 Pauli string 数の実測

`biqbird_quantum.build_qubit_h.build(integ)` で構築した `QubitOperator` の項数：

| 系 | $N_{\mathrm{DVR}}$ | $N_q$ | 1-body Pauli string | 2-body Pauli string | 合計 |
|---|---|---|---|---|---|
| H Step 1 | 3 | 6 | 21 | 13 | 34 |
| H Step 3 | 9 | 18 | 168 | 84 | 252 |
| He Step 6 | 9 | 18 | 168 | 84 | 252 |
| He Step 7 | 11 | 22 | 252 | 124 | 376 |
| He Step 8 | 14 | 28 | 408 | 196 | 604 |

→ **$N_q$ に対し $O(N_q^2)$ 程度**で抑えられる（Gaussian basis なら $O(N_q^4)$ ）。

### 3.5 実装：`build_qubit_h.py`

```python
H = FermionOperator()

# 1-body
for p in range(N):
    for q in range(N):
        for s in (0, 1):
            ip = 2*p + s; iq = 2*q + s
            H += FermionOperator(((ip, 1), (iq, 0)), h_pq[p, q])

# 2-body (DVR collapsed)
for p in range(N):
    for q in range(N):
        for s in (0, 1):
            for sp in (0, 1):
                if p == q and s == sp:
                    continue              # Pauli exclusion
                ip = 2*p + s; iq = 2*q + sp
                H += FermionOperator(
                    ((ip, 1), (iq, 1), (iq, 0), (ip, 0)),
                    0.5 * V_pq[p, q],
                )

qubit_op = jordan_wigner(H)               # OpenFermion does the heavy lifting
```

Action code `1` が生成、`0` が消滅。`jordan_wigner` は §3.1 の置換を機械的に実行し、
類似項を集約して `QubitOperator` (= Σ c_k P_k 形式) を返す。

---

## 4. 粒子数セクター直接構築（メモリ最適化）

### 4.1 動機

$\hat H_{\mathrm{el}}$ は粒子数 $\hat N$ と可換なので、 $N_e$ セクター上で完全に閉じる。
セクター次元は二項係数

$$
D = \binom{N_q}{N_e} = \binom{2 N_{\mathrm{DVR}}}{N_e}
$$

| 系 | $N_q$ | $2^{N_q}$ | $N_e$ | $D$ | $D / 2^{N_q}$ |
|---|---|---|---|---|---|
| He Step 6 | 18 | $2.6 \times 10^5$ | 2 | 153 | $5.8 \times 10^{-4}$ |
| He Step 7 | 22 | $4.2 \times 10^6$ | 2 | 231 | $5.5 \times 10^{-5}$ |
| He Step 8 | 28 | $2.7 \times 10^8$ | 2 | 378 | $1.4 \times 10^{-6}$ |

→ 28 qubit でも **70 万倍** 状態数が削減できる。
古典シミュレータでこのサイズを扱うには必須の最適化。

### 4.2 セクター上行列の構築

OpenFermion の `get_number_preserving_sparse_operator(fermion_op, N_q, N_e)` は、
**FermionOperator を occupation-number basis 上の sparse 行列**として組む。
内部実装は次の流れ（概略）：

1. 全占有パターン $\lbrace |x\rangle : \mathrm{popcount}(x) = N_e \rbrace$ を lex 順で枚挙、 $D$ 個。
2. 各 fermion 単項 $c^\dagger_a c^\dagger_b c_c c_d$ （係数 $\alpha$ ）について、
   ベース $|x\rangle$ に作用させて
   - 占有・非占有のチェック（不可能な作用は 0）
   - JW 符号（ $(-1)^{\lvert\lbrace j: j \lt  a,\ x_j = 1 \rbrace\rvert}$ 等）
   - 結果 $|y\rangle$ の lex インデックスをルックアップ
   - sparse 行列の $(y, x)$ 要素に $\alpha \cdot \mathrm{sgn}$ を加算
3. すべての項を集約して CSR/CSC で返す。

この行列は **$D \times D$ で sparse**（密度 $\sim$ 1-body 項数 / $D$ なので He Step 8 で $\sim 0.1\%$ ）、
メモリ・対角化コストが劇的に下がる。

### 4.3 セクター基底と full Hilbert との対応

`particle_number_basis(N_q, N_e)` は popcount = $N_e$ の整数を昇順に並べる：

```python
indices = np.arange(2**N_q)
popcount = bit_popcount(indices)
sec = np.where(popcount == N_e)[0]   # length D, sorted ascending
```

これは `get_number_preserving_sparse_operator` 内部のセクター順序と
**完全に整合する**（OpenFermion 1.6.x で確認済み）ので、
セクター上の固有ベクトル $v \in \mathbb{C}^D$ を full Hilbert に持ち上げるときは

```python
psi_full = np.zeros(2**N_q, dtype=complex)
psi_full[sec] = v
```

で OK。

### 4.4 実装：`direct_diag.py` の二系統

```python
# Route 1: full JW route  (n_qubits ≲ 18)
H_full = get_sparse_operator(qubit_op, n_qubits)        # 2^N_q × 2^N_q sparse
H_sub  = H_full[sec, :][:, sec]                         # fancy index

# Route 2: sector-direct route  (n_qubits up to ~28)
H_sec  = get_number_preserving_sparse_operator(
            fermion_op, num_qubits=N_q, num_electrons=N_e,
            spin_preserving=False)                       # D × D sparse
```

メモリ使用量実測（He 28 qubit、Step 8）：
- Route 1（full）： $\gt  16$ TB → **不可能**
- Route 2（sector）：8.4 GB ✅

---

## 5. Solver A : 古典直接対角化（reference）

### 5.1 アルゴリズム

DVR セクター行列 $H_{\mathrm{sec}} \in \mathbb{C}^{D \times D}$ （実 Hermitian）の
最低固有値・固有ベクトルを求める：

$$
H_{\mathrm{sec}} v_0 = E_0 v_0, \quad E_0 = \min \mathrm{spec}(H_{\mathrm{sec}})
$$

これは **DVR basis 上の FCI**（電子相関を厳密に取り込んだ完全 CI）と等価。
量子計算結果を比較する古典側基準値（"oracle"）として使う。

| dim $D$ | 推奨 method |
|---|---|
| $\le 4096$ | dense `np.linalg.eigh`（ $O(D^3)$ 、安定） |
| $\gt  4096$ | sparse `scipy.sparse.linalg.eigsh(k=1, which="SA")`（Lanczos） |

### 5.2 Hermite 化

`get_number_preserving_sparse_operator` は理論上 Hermitian だが、
丸め誤差で $H \neq H^*$ となりうる。Lanczos の数値安定性のため

```python
H_sym = 0.5 * (H_sec + H_sec.conj().T)
```

を陽に行う（`direct_diag.lowest_eigenvalue_sector` の中）。

### 5.3 ノルムと収束基準

`eigsh(k=1, which="SA", tol=1e-10)` で 10 桁精度の最低固有値が取れる。
He Step 8（ $D = 378$ ）では dense でも 10 秒（メモリ 8.4 GB）で済むが、
これは $D \times D$ dense 化（ $D=378$ で 1.1 MB）+ Lanczos 内部用ワークの合計が大きい
（OpenFermion の sector マトリクス組立が一時的に 8 GB 程度を使う）。

---

## 6. Solver B : Iterative QPE（量子）

### 6.1 標準 QPE の数式

ユニタリ $U = e^{-i \hat H \tau}$ の固有値 $e^{-2\pi i \phi}$ （ $\phi \in [0,1)$ ）を
$n_{\mathrm{bits}}$ 桁の 2 進小数で測定する。標準 QPE は QFT register
（ $n_{\mathrm{anc}}$ qubit ancilla）を使い

$$
\begin{aligned}
|0\rangle^{\otimes n_{\mathrm{anc}}} |\psi\rangle
&\xrightarrow{H^{\otimes n_{\mathrm{anc}}}}
\frac{1}{\sqrt{2^{n_{\mathrm{anc}}}}}\sum_{m=0}^{2^{n_{\mathrm{anc}}}-1} |m\rangle |\psi\rangle \\
&\xrightarrow{\text{controlled-}U^m}
\frac{1}{\sqrt{2^{n_{\mathrm{anc}}}}}\sum_m e^{-2\pi i m\phi} |m\rangle |\psi\rangle \\
&\xrightarrow{\mathrm{QFT}^\dagger}
|\tilde m\rangle |\psi\rangle, \quad \tilde m \approx 2^{n_{\mathrm{anc}}}\phi
\end{aligned}
$$

メモリ：状態ベクトルは $2^{N_{\mathrm{sys}} + n_{\mathrm{anc}}}$ 次元 → He 28 qubit + 8 bit ancilla で 1 TB
→ **24 GB PC ではアウト**（[`quantum_computing_plan.md`](../../../BiqBird_Doc/quantum_computing_plan.md) §2.3 参照）。

### 6.2 Iterative QPE（iQPE）

ancilla を **1 qubit に削減**し、 $n_{\mathrm{bits}}$ ラウンドの逐次測定 + 古典フィードバックで
同じ精度を得る（Kitaev / Dobšíček 2007）。
本実装は **LSB-first**：bit $b_0$ （LSB）を最初に、 $b_{n_{\mathrm{bits}} - 1}$ （MSB）を最後に測る。

#### 6.2.1 1 ラウンドの量子回路

ラウンド $k$ （ $k = 0, \dots, n_{\mathrm{bits}} - 1$ ）で `power_k = 2^{n_bits - 1 - k}` を使う：

```
   ancilla  ──|0⟩─[H]──•─[Rz(-ω_k)]─[H]──[M]==> b_k
                       │
   system   ──|ψ⟩───────U^{power_k}─────────  (project on b_k branch)
```

数式：

$$
|0\rangle\otimes|\psi\rangle \xrightarrow{H_{\mathrm{anc}}} \tfrac{1}{\sqrt{2}}(|0\rangle + |1\rangle)\otimes|\psi\rangle
$$

$$
\xrightarrow{C\text{-}U^{\mathrm{power}_k}} \tfrac{1}{\sqrt{2}}\bigl(|0\rangle\otimes|\psi\rangle + |1\rangle\otimes U^{\mathrm{power}_k}|\psi\rangle\bigr)
$$

固有状態前提（ $U^k|\psi\rangle = e^{-iEk\tau}|\psi\rangle$ 、 $\theta_k = -E\tau\cdot\mathrm{power}_k$ ）：

$$
= \tfrac{1}{\sqrt{2}}\bigl(|0\rangle + e^{i\theta_k}|1\rangle\bigr)\otimes|\psi\rangle
$$

フィードフォワード位相補正 $R_z(-\omega_k)$ on ancilla（ $|1\rangle$ 側に $e^{-i\omega_k}$ ）：

$$
\to \tfrac{1}{\sqrt{2}}\bigl(|0\rangle + e^{i(\theta_k - \omega_k)}|1\rangle\bigr)\otimes|\psi\rangle
$$

最終 Hadamard：

$$
\xrightarrow{H_{\mathrm{anc}}} \Bigl[\cos\tfrac{\theta_k - \omega_k}{2}|0\rangle + i\sin\tfrac{\theta_k - \omega_k}{2}|1\rangle\Bigr]\otimes|\psi\rangle
$$

測定確率：

$$
P(\text{ancilla} = 0) = \cos^2\!\Bigl(\tfrac{\theta_k - \omega_k}{2}\Bigr)
$$

#### 6.2.2 フィードフォワード補正の導出

$\phi$ が **正確に $n_{\mathrm{bits}}$ 桁の二進小数** $\phi = m/2^{n_{\mathrm{bits}}}$
（ $m = \sum_j b_j 2^j$ ）と仮定する。
ラウンド $k$ で：

$$
\theta_k
= -2\pi\phi \cdot 2^{n_{\mathrm{bits}}-1-k} \bmod 2\pi
= -\pi b_k - 2\pi \sum_{j=0}^{k-1} b_j \cdot 2^{j-k-1} \pmod{2\pi}
$$

導出： $\phi \cdot 2^{n_{\mathrm{bits}}-1-k} = m/2^{k+1} = b_k/2 + \sum_{j\lt k} b_j 2^{j-k-1} + (\text{整数})$
で、整数部分は $2\pi$ 倍してから mod $2\pi$ で消える。

bit $b_0,\dots,b_{k-1}$ は既に測定済みなので、
**第 2 項を打ち消すフィードフォワード**：

$$
\omega_k = -\pi \sum_{j=0}^{k-1} b_j \cdot 2^{j-k} \pmod{2\pi}
$$

これで $\theta_k - \omega_k = -\pi b_k$ 、よって

$b_k = 0$ のとき $P(0) = \cos^2(0) = 1$ 、 $b_k = 1$ のとき $P(0) = \cos^2(-\pi/2) = 0$ 。

固有状態入力では **決定論的に正しい bit を回収する**（厳密一致）。

#### 6.2.3 アルゴリズム擬似コード

```python
def run_iqpe(H, eigenstate, tau, n_bits):
    psi = eigenstate / norm(eigenstate)
    E   = <psi| H |psi>                  # eigenvalue (1 sparse mat-vec)
    bits = []                            # b_0 (LSB) first
    for k in range(n_bits):
        power_k = 2**(n_bits - 1 - k)
        theta_k = -(E * tau * power_k) mod 2π
        omega_k = -π * Σ_{j<k} bits[j] * 2^{j - k}   mod 2π
        delta   = (theta_k - omega_k) / 2
        p_zero  = cos²(delta)
        b_k     = sample_bit(p_zero)            # = round(1 - p_zero)
        bits.append(b_k)
    m     = Σ_k bits[k] * 2^k
    phase = m / 2^n_bits
    energy = 2π * phase / tau               # mod 2π/tau
    return bits, phase, energy
```

総計 **$O(n_{\mathrm{bits}})$ 古典計算 + 1 sparse mat-vec**（最初の $E$ 計算のみ）。
量子コンピュータ実機なら $O(2^{n_{\mathrm{bits}}})$ ゲート（controlled- $U^{2^{n-1}}$ が一番重い）。

#### 6.2.4 spectrum shift（`shifted_for_iqpe`）

iQPE は $\phi \in [0,1)$ を仮定する。
He の最低固有値 $E_0 \approx -2.1$ Ha は負だし、
$|E\tau| \lt  2\pi$ も保証する必要がある。
そこで cheap eigsh で $[E_{\mathrm{min}}, E_{\mathrm{max}}]$ を粗推定して

$$
\hat H' = \hat H + \mathrm{shift}\cdot I, \quad \mathrm{shift} = -E_{\mathrm{min}} + \mathrm{margin}\cdot(E_{\mathrm{max}} - E_{\mathrm{min}})
$$

$$
\tau = \frac{2\pi}{(E_{\mathrm{max}} + \mathrm{shift})(1 + \mathrm{margin})}
$$

として、シフト済み固有値 $E_0^{\mathrm{shift}} = E_0 + \mathrm{shift}$ から $E_0 = E_{\mathrm{iqpe}} - \mathrm{shift}$ で復元。margin = 0.05 を採用し、`shifted_for_iqpe(H, margin=0.05)` を呼ぶ。

#### 6.2.5 シミュレータ実装の二系統

| 関数 | 入力 | コスト | 用途 |
|---|---|---|---|
| `run_iqpe` | exact eigenstate | 1 mat-vec + 古典 | reference / Q1〜Q3 |
| `run_iqpe_general` | 任意状態 | $n_{\mathrm{bits}}$ 個の `expm_multiply` | Q4 以降の VQE 連携 |

**Why fast path?** ナイーブな statevector シミュレーションでは
ラウンド $k$ で `expm_multiply(-i τ · 2^{n-1-k} H, ψ)` を呼ぶ。
$\tau \cdot 2^{n-1} \cdot \|H\|$ が大きいと `expm_multiply` 内部で
Krylov 部分空間の degree $\sim O(\tau \cdot 2^{n-1} \|H\|)$ が必要になり、
He 18 qubit では数千の sparse mat-vec が 1 ラウンドで発生して数十秒かかる。
固有状態前提なら $U^{\mathrm{power}} |\psi\rangle = e^{-i E \tau \cdot \mathrm{power}} |\psi\rangle$
で位相だけ付ければよく、行列積は 0 回（ $\sim 10^{-2}$ 秒）。

iQPE は本質的に位相を bit に分解する **古典計算 + 1 量子位相測定**であり、
固有状態を用意できる状況ではこの fast path が物理的に正しい近似ではなく
**完全に等価**（ $U$ の固有関係を厳密に適用しているので Trotter 誤差すら発生しない）。

### 6.3 解像度

iQPE の最小エネルギー単位（resolution）：

$$
\Delta E = \frac{2\pi}{\tau \cdot 2^{n_{\mathrm{bits}}}}
$$

$n_{\mathrm{bits}} = 14$ 、He 18 qubit（ $\tau \sim 1.5$ ）で $\Delta E \sim 2.5 \times 10^{-4}$ Ha。
3 桁追加で $n_{\mathrm{bits}} = 24$ にすれば $\sim 2.4 \times 10^{-7}$ Ha まで取れるが、
**古典シミュレータ的には何 bit でも瞬時**（fast path のため）。
量子実機では $n_{\mathrm{bits}}$ ごとにコヒーレンス時間が $2 \times$ 必要なので
fault-tolerant 量子コンピュータ前提。

---

## 7. Solver C : Trotter 時間発展（量子）

### 7.1 演算子分割

$\hat H$ の JW 像を 2 つに分ける：

- $\hat T$ : **計算基底で非対角な** Pauli string（kinetic + exchange、 $XX$/$YY$ 含むもの）
- $\hat V$ : **計算基底で対角な** Pauli string（external pot + Coulomb、 $Z$ と $ZZ$ のみ）

§3.3 で見た通り DVR では 2-body 項が完全に対角に落ちるので、
**$\hat T$ は 1-body の $p \neq q$ 項のみ**、しかもバンド幅 $\sim 2 n_{\mathrm{order}} + 1$ 。

### 7.2 2nd-order Strang splitting

$$
U(\Delta t) \approx e^{-i \hat T \Delta t/2} \cdot e^{-i \hat V \Delta t} \cdot e^{-i \hat T \Delta t/2}
$$

局所誤差 $O(\Delta t^3)$ 、大域誤差 $O(\Delta t^2)$ （一般状態に対し）。
**固有状態に作用させる場合は誤差が $O(\Delta t^4)$**：
2nd-order symmetric splitting で奇数次の項が打ち消すため
（[Hatano-Suzuki 1991](https://doi.org/10.1143/JPSJ.62.49) 系）。

### 7.3 量子回路コスト見積もり（量子実機向け）

| 因子 | 演算 | 量子ゲート数 |
|---|---|---|
| $e^{-i V_{\mathrm{ne}}(x_p) \hat n_p \Delta t}$ （1-body 対角）| $R_z$ | $N_q$ |
| $e^{-i V_{pq} \hat n_p \hat n_q \Delta t}$ （2-body 対角）| $R_{zz}$ | $\binom{N_q}{2}$ |
| $e^{-i T_{pq} (X_p X_q + Y_p Y_q) \Delta t / 2}$ （1-body オフ対角）| Givens / fSim | $O(N_q \cdot \text{bandwidth})$ |
| 合計 / step | | $O(N_{\mathrm{DVR}}^2)$ |

これが **DVR + JW でなければ** $O(N^4)$ になる（Gaussian basis の場合）。

### 7.4 シミュレータ実装

`biqbird_quantum.trotter`：

```python
def split_diag_offdiag(H):
    diag = H.diagonal()
    H_off = H - diag(diag)               # off-diagonal part
    return H_off, diag

def trotter2_step(psi, H_off, V_diag, dt):
    psi = expm_multiply(-1j * 0.5 * dt * H_off, psi)        # T half-step
    psi = exp(-1j * dt * V_diag) * psi                       # V full-step (componentwise)
    psi = expm_multiply(-1j * 0.5 * dt * H_off, psi)        # T half-step
    return psi
```

セクター行列を渡す（ $D \sim 100$ 〜 $500$ ）ので、
`expm_multiply` は数 ms で完了し、
$T_{\mathrm{total}} = 5$ a.u. の伝搬を $\Delta t = 0.02$ （250 step）でも 0.1 秒。

### 7.5 観測量

各サンプリング時刻 $t$ で記録：

- **エネルギー**： $\langle \psi(t) | \hat H | \psi(t)\rangle$ （保存量、Trotter 誤差で揺らぐ）
- **オーバーラップ**： $|\langle \psi(0) | \psi(t)\rangle|^2$ （固有状態なら $\equiv 1$ が exact）

Phase Q4 ではこの 2 つの $\Delta t$ 依存を見て **$O(\Delta t^4)$ scaling** を確認した。

### 7.6 高次 Trotter / Q5 への布石

`trotter.py` は 2nd-order のみ実装。
4th-order Suzuki-Trotter は

$$
U_4(\Delta t) = U_2(s\Delta t)^2 \cdot U_2((1-4s)\Delta t) \cdot U_2(s\Delta t)^2, \quad s = \frac{1}{4 - 4^{1/3}}
$$

で組める。Q5（HHG デモ）で精度が必要になったら拡張する。

---

## 8. データフローと JSON スキーマ

### 8.1 データフロー図

```
┌───────────────────┐
│ FEDVRGrid         │  (C++)
│ build_h_pq        │
│ build_V_pq        │
│ nuclear_repulsion │
└─────────┬─────────┘
          │ JSON  (mini_json::Writer 自作)
          ▼
┌───────────────────┐
│ load_integrals.py │ → Integrals(tag, L, N_DVR, x, w, h_pq, V_pq, ...)
└─────────┬─────────┘
          │
          ▼
┌─────────────────────────┐
│ build_qubit_h.py        │
│   FermionOperator       │  (DVR-collapsed 2-body)
│   ↓ jordan_wigner()     │  (OpenFermion)
│   QubitOperator (252+) │
└─────┬───────────────────┘
      │
      ├──────────────────────────────────────────────┐
      │                                              │
      ▼                                              ▼
┌──────────────────────┐               ┌──────────────────────────┐
│ direct_diag.py       │               │ iqpe.py (run_iqpe)        │
│   sector projection  │               │   shifted_for_iqpe        │
│   eigsh / eigh       │               │   eigenstate fast path    │
│   E_FCI, ψ_GS        │               │   bit-by-bit feedback     │
└──────────────────────┘               └──────────────────────────┘
                              ▲                                   │
                              │                                   ▼
                              │                       (E_iqpe ≈ E_FCI)
                              │
                  ┌───────────┴────────────┐
                  │                        │
                  ▼                        │
       ┌──────────────────────┐            │
       │ trotter.py           │            │
       │   split_diag_offdiag │            │
       │   trotter2_step      │   ─────────┘
       │   evolve             │
       └──────────────────────┘
```

### 8.2 JSON スキーマ

`export_integrals.cpp` 生成、`load_integrals.py` 読込：

```jsonc
{
  "model": {
    "tag":        "<string>",        // free-form identifier
    "a":          1.0,               // soft-Coulomb param (固定)
    "L":          10.0,              // half-range [-L, L]
    "n_elements": 2,                 // FEDVR n_elem
    "n_order":    5,                 // FEDVR n_order
    "N_DVR":      9,                 // = n_elements * n_order - 1
    "N_e":        2,                 // 電子数
    "nuclei": [{"Z": 1.0, "X": +0.8}, {"Z": 1.0, "X": -0.8}],
    "E_nn":       0.529999           // soft-Coulomb 核間反発
  },
  "x": [...],                        // length N_DVR, DVR ノード座標
  "w": [...],                        // length N_DVR, DVR 重み
  "h_pq": [[..., ...], ...],         // N_DVR × N_DVR, 1電子積分（dense）
  "V_pq": [[..., ...], ...]          // N_DVR × N_DVR, 2電子 kernel（dense）
}
```

すべての浮動小数点は `std::setprecision(17)`（IEEE 754 double を完全往復）で出力。

---

## 9. 段階的検証結果（Phase Q1〜Q4）

### 9.1 Phase Q1 : H 原子（量子計算 Hello World）

| Step | $L$ | $N_e^{\mathrm{elem}}$ | $n_{\mathrm{order}}$ | $N_{\mathrm{DVR}}$ | $N_q$ | sector dim | $E_{\mathrm{classical}}$ [Ha] | $E_{\mathrm{direct}}$ [Ha] | $E_{\mathrm{iqpe}}$ [Ha] | iQPE \|Δ\| [Ha] |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 5  | 1 | 4 | 3 | 6  | 6  | $-0.8871244856$ | $-0.8871244856$ | n/a | (粗すぎ) |
| 3 | 10 | 2 | 5 | 9 | 18 | 18 | $-0.6360714341$ | $-0.6360714341$ | $-0.6387759174$ | $2.7 \times 10^{-3}$ |

連続極限値（ $L=20, n=8$ ）： $E_0 \approx -0.66977$ Ha。
Step 1 が $-0.887$ と深いのは $L=5$ で電子が箱に押し込まれるため（境界効果）。

JW 機構の検証：1電子セクターでの直接対角化が $h_{pq}$ 直接対角化と機械精度（ $\sim 10^{-15}$ ）で一致。
iQPE は 12-bit resolution（ $\sim 1.1 \times 10^{-2}$ Ha）以内で direct diag と一致。

### 9.2 Phase Q2 : He 原子（FCI 級）

| Step | $L$ | $N_e^{\mathrm{elem}}$ | $n_{\mathrm{order}}$ | $N_{\mathrm{DVR}}$ | $N_q$ | sector dim | $E_{\mathrm{FCI}}$ [Ha] | $E_{\mathrm{iqpe}}$ [Ha] | iQPE \|Δ\| | RSS |
|---|---|---|---|---|---|---|---|---|---|---|
| 6 | 10 | 2 | 5 | 9  | 18 | 153 | $-2.1103945808$ | $-2.1104041667$ | $9.6 \times 10^{-6}$ | 218 MB |
| 7 | 10 | 2 | 6 | 11 | 22 | 231 | $-2.1959750914$ | $-2.1959926359$ | $1.8 \times 10^{-5}$ | 375 MB |
| 8 | 10 | 3 | 5 | 14 | 28 | 378 | $-1.9818514741$ | $-1.9818785182$ | $2.7 \times 10^{-5}$ | 8.4 GB |

参照値（[`../README.md`](../README.md)）：
- Octopus FCI（ $a=1$, $L=8$, $dx=0.1$ ）: $-2.23826$ Ha
- RHF 連続極限： $-2.22421$ Ha
- 相関エネルギー： $\sim 0.014$ Ha

**観察**：
- iQPE（14 bit、resolution $\sim 5\times 10^{-4}$ Ha）は全 Step で $E_{\mathrm{FCI}}$ と一致
- Step 7 が一番深い（ $-2.196$ Ha）： $n=6$ で要素境界少なく FEDVR の指数収束が効く
- Step 8 が逆に浅い（ $-1.982$ Ha）：要素数 3 で要素境界が増え、ブリッジ点の精度劣化

### 9.3 Phase Q3 : H₂ ポテンシャル曲線（解離問題の治癒）

設定： $L=10$, $N_e^{\mathrm{elem}}=2$, $n_{\mathrm{order}}=5 \Rightarrow N_{\mathrm{DVR}}=9$, $N_q=18$, sector dim = 153

| $R$ [a.u.] | $E_{\mathrm{FCI}}$ | $E_{\mathrm{iqpe}}$ | $E_{\mathrm{nn}}$ | $E_{\mathrm{tot}}^{\mathrm{Q}}$ | $E_{\mathrm{tot}}^{\mathrm{RHF}}$ | $\Delta E_{\mathrm{corr}}$ |
|---|---|---|---|---|---|---|
| 1.0 | $-2.0600$ | $-2.0600$ | 0.7071 | $-1.353$ | $-1.394$ | $-0.042$ |
| 1.6 | $-2.0375$ | $-2.0375$ | 0.5300 | $-1.507$ | $-1.425$ | $+0.082$ |
| 3.0 | $-1.7734$ | $-1.7735$ | 0.3162 | $-1.457$ | $-1.311$ | $+0.146$ |
| 6.0 | $-1.5942$ | $-1.5941$ | 0.1644 | $-1.430$ | $-1.121$ | **$+0.309$** |

$\Delta E_{\mathrm{corr}} = E_{\mathrm{tot}}^{\mathrm{RHF}} - E_{\mathrm{tot}}^{\mathrm{Q}}$ （正なら量子が下回る）。

**観察**：
- $R = 1.0$ で量子が上に位置するのは古典 RHF（ $N_{\mathrm{DVR}}=79$ の細かいグリッド）vs
  量子（ $N_{\mathrm{DVR}}=9$ の粗いグリッド）のグリッド差
- $R \ge 1.6$ で量子が一貫して下回り、**$R = 6.0$ では $0.31$ Ha の差**：
  これが教科書的な「RHF 解離破綻 vs FCI/quantum の正しい解離」のデモ
- 解離極限 $2 E_0^{\mathrm{H}} \approx -1.34$ Ha に対し量子が $-1.43$ （やや深め=$N_{\mathrm{DVR}}=9$ の粗さで H 原子が深い）、
  RHF は $-1.12$ で大きく上に詰まる

### 9.4 Phase Q4 : Trotter 時間発展（GS のエネルギー保存）

設定：He Step 6（18 qubit）の基底状態を $T_{\mathrm{total}} = 5$ a.u. 伝搬。
2nd-order Strang splitting:

$$
U(\Delta t) = e^{-i T \Delta t/2} \cdot e^{-i V \Delta t} \cdot e^{-i T \Delta t/2}
$$

| $\Delta t$ | n_steps | $\max \lvert \Delta E \rvert$ [Ha] | final overlap | scaling 比 |
|---|---|---|---|---|
| 0.500 | 10  | $2.91 \times 10^{-4}$ | 0.99985 | — |
| 0.200 | 25  | $7.07 \times 10^{-6}$ | 0.99999 | 41× |
| 0.100 | 50  | $4.39 \times 10^{-7}$ | 1.00000 | 16× |
| 0.050 | 100 | $2.74 \times 10^{-8}$ | 1.00000 | 16× |
| 0.020 | 250 | $7.00 \times 10^{-10}$ | 1.00000 | 39× |

**観察**：
- $\Delta t \to \Delta t/2$ で誤差が $\sim 16$ 倍縮小 → **$O(\Delta t^4)$ scaling**
  （symmetric 2nd-order Trotter が固有状態に作用すると偶数次のみ生き残る）
- Final overlap = 1.000000 で **状態自体も復元**（global phase は別、|⟨ψ₀|ψ(T)⟩|² なので位相は無関係）
- 量子化学 Trotter で「GS が定常状態として保たれる」ことを確認 → Q5（外部レーザー）への基盤完成

---

## 10. 既存 FD/FEDVR・連続極限値との比較

### 10.1 同じ物理問題か？

`fedvr::build_h_pq(grid, nuclei)` と `fedvr::build_V_pq(grid)` は
1 行も書き換えずに `export_integrals.cpp` から呼ばれており、
量子計算側で扱う Hamiltonian は **古典 FEDVR の RHF 計算と数値的に同一**。
違いは：

| | 古典 RHF (`hf_fedvr.hpp`) | 量子計算 (`biqbird_quantum`) |
|---|---|---|
| 波動関数の表現 | 単一空間軌道 + L²正規化 | 全 Slater 行列式の重ね合わせ |
| 取り込む相関 | なし（mean-field） | 全部（FCI in DVR basis） |
| 計算量 | $O(N^3)$ per SCF iter | $O(D^2)$ per Lanczos iter（ $D = \binom{N_q}{N_e}$ ） |

### 10.2 既存連続極限値（[`../README.md`](../README.md)）

| | 連続極限値 [Ha] | 出典 |
|---|---|---|
| H | $-0.66977$ | FEDVR $L=20, n=12$ |
| He（RHF）| $-2.22421$ | FEDVR $L=20, n=12$ |
| He（FCI）| $-2.23826$ | Octopus $L=8, dx=0.1$ |
| H₂（RHF, $R_e=1.6$ ）| $-1.42507$ | FEDVR $L=20, n=8$ |

量子計算側（grid が粗い）はこれより浅いが、連続極限への収束方向は同じ。
Phase Q4.5（任意）として grid を細かくしてのスキャンは将来課題。

---

## 11. ビルド・実行手順

### 11.1 C++ JSON エクスポータ

```bash
cd Lab/BiqBird/Sample/1D/Quantum
mkdir -p out data
EIGEN=/opt/homebrew/include/eigen3
g++ -std=c++17 -O2 -I"$EIGEN" -I"../FEDVR" \
    export_integrals.cpp -o out/export_integrals
```

実行例：

```bash
# H 6 qubit (Phase Q1 Step 1)
./out/export_integrals --L 5 --Ne 1 --n 4 --Z 1 --Nelec 1 \
    --tag h_step1 --out data/h_step1.json

# He 28 qubit (Phase Q2 Step 8)
./out/export_integrals --L 10 --Ne 3 --n 5 --Z 2 --Nelec 2 \
    --tag he_step8 --out data/he_step8.json

# H_2 (Phase Q3, R = 1.6 a.u.)
./out/export_integrals --L 10 --Ne 2 --n 5 --Nelec 2 \
    --H2 --R 1.6 --tag h2_R1_6 --out data/h2_R1_6.json
```

### 11.2 Python 側（uv）

```bash
cd Lab/BiqBird/Sample/1D/Quantum
uv sync                                   # .venv 作成 & 依存解決

# Phase Q1
uv run python scripts/q1_step1_h_6qubit.py
uv run python scripts/q1_step3_h_18qubit.py

# Phase Q2 (He 18/22/28 qubit)
uv run python scripts/q2_he.py he_step6
uv run python scripts/q2_he.py he_step7
uv run python scripts/q2_he.py he_step8

# Phase Q3 (H_2 scan)
uv run python scripts/q3_h2_scan.py

# Phase Q4 (Trotter Δt scan)
uv run python scripts/q4_trotter_gs.py
```

### 11.3 ビルド・依存関係

| 依存 | バージョン | 用途 |
|---|---|---|
| Eigen3 | $\ge 3.4$ | C++ 行列演算 |
| `nlohmann/json` | 不使用（`mini_json::Writer` 自作） | — |
| Python | $\ge 3.11$ | uv 推奨 |
| `numpy` | $\ge 1.26$ | 数値配列 |
| `scipy` | $\ge 1.13$ | sparse, eigsh, expm_multiply |
| `openfermion` | $\ge 1.6$ | FermionOperator, JW, sector projector |
| `qiskit`, `qiskit-aer` | $\ge 1.2 / 0.15$ | iQPE のリファレンス量子回路（§12） |

---

## 12. Qiskit による量子回路レベルでの再構成（Phase Q5）

§6 の iQPE プロトコルを Qiskit Aer のステートベクトルシミュレータ上で量子回路として記述し、§6.2.5 の eigenstate fast path（古典）と同じ結果が得られることを確認する。これにより、§0.3 (A) で固有値情報を解析的に利用している `iqpe.run_iqpe` が、controlled- $U^{2^k}$ を ancilla 制御として実際に作用させて測定するルートと数値的に等価であることを示す。

### 12.1 量子回路の構造（H 6 qubit）

[`scripts/q5_qiskit_reference.py`](scripts/q5_qiskit_reference.py) は H 6 qubit（system 6 + ancilla 1 = 計 7 qubit）について §6.2.1 の回路を 12 ラウンド分組み立てる。1 ラウンドの構造を以下に示す。

```
ラウンド k = 0..n_bits-1 (LSB first):

  sys[0..5]  ──|ψ_GS⟩─────────────────────│U^{2^{n-1-k}}│──────────────
                                                │
                                                ●  (control on ancilla)
                                                │
  anc[0]     ──|0⟩──[H]──────────────────[●]──[Rz(-ω_k)]──[H]──[M]==> b_k
                                                                     │
                                                              古典で ω_{k+1} を計算
```

主な実装要素は次の通り。

1. **状態準備**: `qc.initialize(psi_full, sys_reg)` で直接対角化により得た固有ベクトルを system register に焼き付ける（§0.3 (A) で述べた固有状態前提）。
2. **ancilla 初期化**: `qc.h(anc_reg[0])` で $|+\rangle$ を準備。
3. **controlled-** $U^{\mathrm{power}}$ : シフト済み Hamiltonian の対角化 $H = V \mathrm{diag}(E_n) V^\dagger$ から $V \mathrm{diag}(e^{-i E_n \tau \cdot \mathrm{power}}) V^\dagger$ を構成し、`UnitaryGate` として `.control(1)` で ancilla 制御化する。実機向けに展開する場合はこのユニタリを §7 の Trotter 分解で Pauli string ごとのゲート列に置き換える。
4. **フィードフォワード**: `qc.rz(-omega_k, anc_reg[0])` で §6.2.2 の閉形式 $\omega_k$ を適用。
5. **測定**: ancilla に Hadamard をかけてから計算基底測定。
6. **シミュレーション**: `transpile` で AerSimulator が解釈できる基本ゲートに分解し、4096 shots で $P(0)$ を統計推定し $b_k$ を決定する。

### 12.2 数値結果

`uv run python scripts/q5_qiskit_reference.py` の実行結果（H Step 1, $n_{\mathrm{bits}} = 12$ ）：

| $k$ | power | $P(0)$ qiskit (4096 shots) | $P(0)$ classical (closed form) | $b_k$ qiskit |
|---|---|---|---|---|
| 0 | 2048 | 0.8687 | 0.8641 | 0 |
| 1 | 1024 | 0.0364 | 0.0352 | 1 |
| 2 |  512 | 0.9907 | 0.9911 | 0 |
| 3 |  256 | 0.0012 | 0.0022 | 1 |
| 4 |  128 | 0.0000 | 0.0006 | 1 |
| 5 |   64 | 0.0000 | 0.0001 | 1 |
| 6 |   32 | 1.0000 | 1.0000 | 0 |
| 7 |   16 | 0.0000 | 0.0000 | 1 |
| 8 |    8 | 1.0000 | 1.0000 | 0 |
| 9 |    4 | 1.0000 | 1.0000 | 0 |
| 10 |   2 | 1.0000 | 1.0000 | 0 |
| 11 |   1 | 1.0000 | 1.0000 | 0 |

復元結果：

```
bits classical (LSB..)  = [0, 1, 0, 1, 1, 1, 0, 1, 0, 0, 0, 0]
bits qiskit    (LSB..)  = [0, 1, 0, 1, 1, 1, 0, 1, 0, 0, 0, 0]
phase classical         = 0.0454101562
phase qiskit            = 0.0454101562
E classical iqpe        = -0.8868100681 Ha
E qiskit    iqpe        = -0.8868100681 Ha
E direct diag           = -0.8871244856 Ha
|E_qiskit - E_classical_iqpe| = 0.000e+00
```

12 ラウンドすべてで $b_k$ が一致し、復元された位相とエネルギーも一致する。`run_iqpe`（§6.2.5）の解析計算と、Qiskit 上で controlled- $U^{2^k}$ を実際に作用させて測定するルートが、固有状態入力に対して同じ結果を返すことが確認できる。

### 12.3 量子実機向けのリソース見積もり

§12.1 の回路を物理的な量子コンピュータ上で実行する場合に必要となるリソースを、He Step 6（system 18 qubit + ancilla 1 qubit）を例として見積もる。

| 項目 | 値 |
|---|---|
| 必要 logical qubit 数 | 19 |
| 1 step あたりの Trotter ゲート数（§7.3）| $O(N_q^2) = 18^2 \approx 320$ |
| 1 ラウンドの controlled- $U^{\mathrm{power}}$ に必要な Trotter step 数 | $\tau \cdot \mathrm{power} / \Delta t$ （ $\tau = 1.5$, power $= 2^{13}$, $\Delta t = 0.05$ で $\sim 5 \times 10^5$ ）|
| 1 ラウンドの累積ゲート数 | $\sim 1.6 \times 10^8$ （最大 power のラウンド）|
| 全 14 ラウンドの累積ゲート数 | $\sim 3 \times 10^8$ |
| 必要なコヒーレンス時間 | $T_{\mathrm{gate}} \times 3 \times 10^8 \sim 30$ 秒（ $T_{\mathrm{gate}} = 100$ ns 仮定）|

現状のノイズ持ち量子プロセッサ（ $T_2 \sim 100$ μs オーダー）では上記のゲート列を保てないため、誤り訂正符号で論理 qubit を構成した量子コンピュータ上での実行が前提となる。

§3.4 の Pauli string 数を Gaussian basis 系（ $O(N_q^4)$ 個）と比較すると、DVR + JW では概ね $1/200$ 程度に収まる。Trotter ゲート数は Pauli string 数に比例するため、DVR 採用は実機向けゲート数のオーダーを 1 桁以上下げる効果を持つ。

### 12.4 Qiskit から実機ジョブとして投げる場合

`scripts/q5_qiskit_reference.py` で構築した `QuantumCircuit` は `qiskit_ibm_runtime` 経由でそのまま実機ジョブとして送信できる形式である。ただし上述のゲート数を全 14 ラウンド分は実用的なジョブサイズに収まらない。1 ラウンドのみ（最小 power $= 1$ ）の限定実行であれば数百ゲートで済むが、誤り訂正なしの場合 shot ノイズによって測定値が確率的に揺らぎ、bit 推定の確度はゲート忠実度に依存する。本リポジトリでは実機ジョブの送信は行わず、Qiskit Aer 上での量子回路シミュレーションをもって §0.3 (A) との整合性確認とする。

---

## 13. 参考文献

### Jordan-Wigner / 量子化学の量子計算

- P. Jordan, E. Wigner, "Über das Paulische Äquivalenzverbot",
  *Z. Phys.* **47**, 631 (1928).
- McClean *et al.*, "OpenFermion: The Electronic Structure Package for Quantum Computers",
  *Quantum Sci. Technol.* **5**, 034014 (2020).
  https://github.com/quantumlib/OpenFermion

### DVR / FEDVR + 量子化学

- T. N. Rescigno, C. W. McCurdy,
  "Numerical grid methods for quantum-mechanical scattering problems",
  *Phys. Rev. A* **62**, 032706 (2000).
- Babbush *et al.*,
  "Encoding electronic spectra in quantum circuits with linear T complexity",
  *Phys. Rev. X* **8**, 011044 (2018).
  → grid/DVR basis 量子化学のベンチマーク。
- Babbush *et al.*,
  "Low-depth quantum simulation of materials",
  *Phys. Rev. X* **8**, 011044 (2018).
  → DVR Trotter のゲート数評価。

### Quantum Phase Estimation

- A. Y. Kitaev, "Quantum measurements and the Abelian Stabilizer Problem",
  arXiv:quant-ph/9511026 (1995).
- M. Dobšíček *et al.*,
  "Arbitrary accuracy iterative quantum phase estimation",
  *Phys. Rev. A* **76**, 030306(R) (2007).
- A. Aspuru-Guzik *et al.*,
  "Simulated Quantum Computation of Molecular Energies",
  *Science* **309**, 1704 (2005).

### Trotter / Suzuki

- M. Suzuki, "General theory of fractal path integrals
  with applications to many-body theories and statistical physics",
  *J. Math. Phys.* **32**, 400 (1991).
- N. Hatano, M. Suzuki,
  "Finding exponential product formulas of higher orders",
  *Lect. Notes Phys.* **679**, 37 (2005).

### 本リポジトリ内の関連文書

- 古典 FEDVR 実装 : [`../FEDVR/README.md`](../FEDVR/README.md)
- 古典 FD 実装 : [`../FD/README.md`](../FD/README.md)
- 1D 全体規約 : [`../README.md`](../README.md)
