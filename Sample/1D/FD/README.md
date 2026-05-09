# 1-D H/He atom — Finite Difference (FD) implementation

有限差分（2次中心差分）+ 4次ルンゲクッタによる **虚時間発展** で、
1次元ソフトクーロンモデルにおける H 原子・He 原子の電子基底状態を求めるサンプル。

物理ハミルトニアン（共通定義）は親ディレクトリ
[`../README.md`](../README.md) を参照。

## ファイル一覧

| ファイル | 内容 |
|---|---|
| `H_GS.cpp`  | 1電子（H 原子）。スタンドアロンの実装 |
| `He.hpp`    | 軌道クラス・ハミルトニアン・Hartree・RK4 を提供するヘッダ |
| `He_GS.cpp` | 閉殻 He（2電子）。`He.hpp` を利用する `main` |

## ハミルトニアン

$$ \hat{H}_{\rm H} = -\frac{1}{2}\frac{d^2}{dx^2} - \frac{1}{\sqrt{x^2+1}} $$

$$ \hat{H}_{\rm He} = \sum_{i=1}^{2}\left(-\frac{1}{2}\nabla_i^2 - \frac{2}{\sqrt{x_i^2+1}}\right) + \frac{1}{\sqrt{(x_1-x_2)^2+1}} $$

閉殻 He は単一空間軌道 $\psi(x)$ について

$$ E[\psi] = 2 \langle\psi|\hat{h}|\psi\rangle + \iint |\psi(x_1)|^2 w(x_1,x_2) |\psi(x_2)|^2 dx_1 dx_2 $$

を最小化する形で解く。

## 数値手法

- 空間離散化：等間隔グリッド、2次中心差分（コメントに4次差分の試行あり）
- 時間発展：虚時間 $\tau$ への置換（ $t \to -i\tau$ ） → $\partial_\tau \psi = -\hat{H}\, \psi$ を **RK4** で進行
- 各ステップ後に L²正規化

## ビルド・実行例

```bash
mkdir -p out
g++ -std=c++17 -O2 H_GS.cpp  -o out/h_gs   && ./out/h_gs
g++ -std=c++17 -O2 He_GS.cpp -o out/he_gs  && ./out/he_gs
```

## 参考パラメータ

| | x_range (L/2) | dx  | dτ   | max_itr |
|---|---|---|---|---|
| H  | 15 | 0.2 | 0.05 | 10000 |
| He | 20 | 0.4 | 0.20 |  1000 |

## 検証結果

He 原子（ $x_{\rm range} = 20$ , $dx = 0.4$ ）：

| 量 | 値 [a.u.] |
|---|---|
| $E_{\rm tot}$ | -2.22837 |
| $E_1 = 2 \langle \psi \lvert \hat{h} \rvert \psi \rangle$ | -2.94721 |
| $E_2$ (Hartree) | 0.71884 |
| 収束反復数 | 239 |

文献値（Octopus FCI 厳密解 $-2.23826$ Ha）から相関エネルギー分 $\sim 0.014$ Ha だけ高い、
Hartree 近似として妥当な値。FEDVR 実装（[`../FEDVR/`](../FEDVR/)）とも $\sim 0.004$ Ha の精度で一致。
