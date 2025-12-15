# -*- coding: utf-8 -*-
"""
小説テキストを学習し返答するチャットボット（Transformer版）

このプログラムは、ローカルに保存した小説テキストファイルを読み込み、
Transformerニューラルネットワークで文章の特徴を学習。
学習後、質問に対して小説の文体で応答を生成。

【主な機能】
- 複数の小説ファイルの自動読み込み
- 学習した文体での文章生成
"""

import os
import re
import math
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pickle
from tqdm import tqdm

# time計測用（最小）
import time


# ============================================================
# ハイパーパラメータ（学習の設定値）
# ============================================================
NOVEL_DIR = "./aozorabunko"     # 小説ファイルのディレクトリ。プロジェクト内にaozorabunkoフォルダを置く
EMBEDDING_DIM = 512             # 文字を数値ベクトルに変換する際の次元数（Transformerは大きめ推奨）
NUM_HEADS = 8                   # Attentionヘッドの数（EMBEDDING_DIMの約数である必要あり）
NUM_LAYERS = 6                  # Transformerの層の数（深いほど複雑なパターンを学習可能）
FF_DIM = 2048                   # Feed-Forward層の中間次元（通常はEMBEDDING_DIMの4倍）
BATCH_SIZE = 192                # 一度に学習するデータのまとまり数（Transformerはメモリ使用量が多いため小さめ）
EPOCHS = 5                      # 全データを何回繰り返し学習するか
LEARNING_RATE = 0.0001          # 学習の速度（Transformerは小さめの学習率が安定）
MAX_LENGTH = 100                # 一度に処理する文字列の最大長
POS_ENCODING_MAX_LEN = 1000     # 位置エンコーディングが対応できる最大長
MODEL_PATH = "novel_transformer_model.pth"        # 学習済みモデルの保存ファイル名
VOCAB_PATH = "vocabulary_transformer.pkl"         # 語彙データの保存ファイル名


class NovelDataset(Dataset):
    """
    小説テキストを、Transformer の学習に使う
    「入力文字列 → 次の文字列」のペアに変換する Dataset クラス。

    ──────────────────────────────────────
    ■ 超・図解：どうやって学習データを作るの？
    テキスト： ABCDEFGHI ...

    max_length = 5 のとき：

      i=0 → 入力: ABCDE → 正解: BCDEF
      i=1 → 入力: BCDEF → 正解: CDEFG
      i=2 → 入力: CDEFG → 正解: DEFGH
      ...

    👉 Transformer に「連続した文章の予測」を学ばせるための仕組み
    ──────────────────────────────────────
    """

    def __init__(self, text, vocab, max_length=MAX_LENGTH):
        self.text = text              # 生テキスト（巨大でも1本だけ保持）
        self.vocab = vocab            # 文字→ID の辞書
        self.max_length = max_length  # 1サンプルあたりの文字数

        # 何個のサンプルが作れるかを計算
        # 例）len(text)=1000, max_length=200 → 1000-200-1 = 799 サンプル
        self.dataset_length = max(0, len(self.text) - self.max_length - 1)


    def create_sequences(self, text):
        """
        テキストを 1 文字ずつずらしながら
        「入力100文字 → 次の100文字」を作成する。
        """

        sequences = []
        text_length = len(text)
        max_len = self.max_length

        # 例：100文字ずつスライドしてペアを作成
        for i in range(text_length - max_len - 1):
            # 入力文字列
            input_seq = text[i : i + max_len]
            # 正解の次の文字列
            target_seq = text[i + 1 : i + max_len + 1]
            # ペアとして記録
            sequences.append((input_seq, target_seq))
        return sequences



    def __len__(self):
        """サンプルの総数を返す（DataLoader がバッチ回数を決めるのに使う）"""
        return self.dataset_length


    def __getitem__(self, idx):
        """
        idx 番目の (入力, 正解) ペアを作って返す。

        実装ポイント：
        - 事前に文字列を全部作って保存しておくのではなく
        - ここで初めて text からスライスして作る
        """
        """
        指定 index の (入力文字列, 正解文字列) を取り出し、
        文字 → ID → Tensor に変換して返す。

        ────────────────────────────────
        ■ 超・図解：文字を ID に変換する流れ

        入力文字列: 「探偵とは」

        文字 → ID 変換
          '探' → 125
          '偵' → 356
          'と' → 22
          'は' → 31

        → テンソル化
          tensor([125, 356, 22, 31])

        Transformer はこの ID を「単語のように」処理する。
        ────────────────────────────────
        """

        # 1. 文字列スライスの開始位置・終了位置を決める
        start = idx
        end = idx + self.max_length

        # 2. 生テキストから「入力」と「次の文字列（正解）」を切り出す
        input_seq = self.text[start:end]        # 長さ max_length
        target_seq = self.text[start + 1:end + 1]  # 1文字右シフト

        # 3. 文字列 → ID リストに変換
        input_ids = []
        for char in input_seq:
            char_id = self.vocab.get(char, self.vocab["<UNK>"])#UNKとは、AIが知らない文字のこと。AI が知らない文字で止まらず、学習・推論を続けるための保険
            input_ids.append(char_id)

        target_ids = []
        for char in target_seq:
            char_id = self.vocab.get(char, self.vocab["<UNK>"])
            target_ids.append(char_id)

        # 4. PyTorch Tensor に変換して返す
        return torch.tensor(input_ids), torch.tensor(target_ids)



class PositionalEncoding(nn.Module):
    """
    位置エンコーディング（Positional Encoding）

    ───────────────────────────────────────
    ■ なぜ必要なの？
      Transformer は「全部の文字を同時に」見るので、そのままだと文字の順番を理解できない。

      例：
          「私は学生です」
          「学生私はです」
      → どちらも同じ文字でも、意味が違う！

      そこで、
      ★ 文字が「何番目にあるか」を示す位置情報 を sin/cos の波形で作り、ベクトルに足し込む。
    ───────────────────────────────────────
    """

    def __init__(self, d_model, max_len=5000):
        """
        Args:
            d_model (int): 埋め込みベクトルの次元数
            max_len (int): 対応できる最大の系列長
        """
        super().__init__()

        # ─────────────────────────────
        # 位置 0,1,2,3,... の番号を作る
        # 形状：[max_len, 1]
        # ─────────────────────────────
        position = torch.arange(max_len).unsqueeze(1)

        # ─────────────────────────────
        # sin/cos の波の周期を決める
        # 次元ごとに異なる周期にする（位置を区別するため）
        # ─────────────────────────────
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
        )

        # 位置エンコーディングの保存先
        pe = torch.zeros(max_len, d_model)

        # 偶数次元：sin 波
        pe[:, 0::2] = torch.sin(position * div_term)

        # 奇数次元：cos 波
        pe[:, 1::2] = torch.cos(position * div_term)

        # モデルに保存するが、学習対象ではない（固定値）
        self.register_buffer('pe', pe)


    def forward(self, x):
        """
        位置エンコーディングをベクトルに加算する。

        Args:
            x: [batch_size, seq_len, d_model]

        Returns:
            x + pe : 位置情報を含んだベクトル
        """

        # 現在の入力シーケンス長
        seq_len = x.size(1)

        # 必要な位置エンコーディングだけ取り出す → [1, seq_len, d_model]
        pe = self.pe[:seq_len].unsqueeze(0)

        # ─────────────────────────────
        # x + pe の形状
        #   x : [batch_size, seq_len, d_model]
        #   pe: [1,         seq_len, d_model]
        #
        #  → PyTorch のブロードキャストで、自動的に
        #     各バッチに同じ pe を足してくれる。
        # ─────────────────────────────
        return x + pe


class NovelTransformer(nn.Module):
    """
    小説の文体を学習するTransformerニューラルネットワーク
    
    
    【構造】
    1. Embedding層: 文字ID → 密なベクトル表現に変換
    2. Positional Encoding: 位置情報を付与
    3. Transformer Encoder層（複数層）:
       - Multi-Head Self-Attention: 文字間の関連性を学習
       - Feed-Forward: さらに特徴を抽出
    4. 全結合層: 次に来る文字を予測
    
    【Multi-Head Attentionとは】
    複数の「視点」（ヘッド）で文章を分析する仕組み。
    例えば8ヘッドなら：
    - Head1: 文法的な関係を学習
    - Head2: 意味的な関係を学習
    - Head3: 長距離の依存関係を学習
    ...など、それぞれが異なるパターンを捉えます。
    """
    
    def __init__(self, vocab_size, embedding_dim, num_heads, num_layers, ff_dim, max_len=MAX_LENGTH):
        """
        モデルの初期化
        
        Args:
            vocab_size (int): 語彙のサイズ（ユニークな文字の総数）
            embedding_dim (int): 埋め込みベクトルの次元数
            num_heads (int): Attentionヘッドの数
            num_layers (int): Transformerの層の数
            ff_dim (int): Feed-Forward層の中間次元
            max_len (int): 処理できる最大文字列長
        """
        super(NovelTransformer, self).__init__()
        
        # Embedding層：文字ID（整数）を密なベクトル（実数の配列）に変換
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        
        # 位置エンコーディング：文字の順序情報を付与
        self.pos_encoder = PositionalEncoding(embedding_dim, max_len=POS_ENCODING_MAX_LEN)
        
        # TransformerEncoderLayer: Self-Attention + Feed-Forward のセット
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,      # d_model: 入出力の次元数（embedding_dimと同じ）
            nhead=num_heads,            # nhead: Attentionヘッドの数（embedding_dimはnheadで割り切れる必要あり）
            dim_feedforward=ff_dim,     # dim_feedforward: Feed-Forward層の中間次元（通常はd_modelの4倍）
            dropout=0.1,                # dropout: 過学習を防ぐためのドロップアウト率
            activation='gelu',          # 活性化関数（GELUはTransformerで一般的）
            batch_first=True            # バッチを最初の次元にする
        )
        
        # TransformerEncoder: 上記のlayerをnum_layers個積み重ねる
        # 層が深いほど複雑なパターンを学習できる
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        
        # 全結合層（線形変換）：Transformerの出力を語彙サイズに変換
        # embedding_dim個の値 → vocab_size個の値（各文字の確率）
        self.fc = nn.Linear(embedding_dim, vocab_size)
        
        # Dropout層：過学習防止
        self.dropout = nn.Dropout(0.1)
        
        self.embedding_dim = embedding_dim
    
    def generate_square_subsequent_mask(self, sz):
        """
        因果的マスク（Causal Mask）の生成
        
        【重要】未来の文字を見ないようにするマスク
        
        文章生成では「現在位置より後の文字」を見てはいけません。
        例：「探偵と」から「は」を予測する時、「は何か」を見てはダメ。
        
        マスクの例（4文字の場合）:
        [[0, -inf, -inf, -inf],   ← 位置0は位置0のみ参照可能
         [0,    0, -inf, -inf],   ← 位置1は位置0,1を参照可能
         [0,    0,    0, -inf],   ← 位置2は位置0,1,2を参照可能
         [0,    0,    0,    0]]   ← 位置3は全て参照可能
        
        -infの部分はAttention計算で無視されます。
        
        Args:
            sz (int): 系列長
        
        Returns:
            形状 [sz, sz] のマスク
        """
        mask = torch.triu(torch.ones(sz, sz) * float('-inf'), diagonal=1)
        return mask
    
    def forward(self, x):
        """
        順伝播：入力データをモデルに通して出力を得る
        
        【処理の流れ】
        入力文字ID → 埋め込み → 位置エンコーディング → Transformer → 全結合 → 各文字の予測スコア
        
        Args:
            x (Tensor): 入力データ（形状: [バッチサイズ, 文字列長]）
        
        Returns:
            Tensor: 予測出力（形状: [バッチサイズ, 文字列長, 語彙サイズ]）
        """
        # 1. 文字IDを埋め込みベクトルに変換
        # [batch_size, seq_len] → [batch_size, seq_len, embedding_dim]
        embedded = self.embedding(x) * math.sqrt(self.embedding_dim)
        # √embedding_dim を掛けるのは、Transformer論文での標準的なスケーリング
        
        # 2. Dropoutを適用（過学習防止）
        embedded = self.dropout(embedded)
        
        # 3. 位置エンコーディングを加算
        # 各文字に「何番目の文字か」という情報を付与
        embedded = self.pos_encoder(embedded)
        
        # 4. 因果的マスクを生成（未来を見ないようにする）
        seq_len = x.size(1)
        mask = self.generate_square_subsequent_mask(seq_len).to(x.device)
        
        # 5. Transformerで文脈を考慮した処理
        # Multi-Head Attentionで全文字間の関連性を学習
        # Feed-Forwardでさらに特徴を抽出
        # これを複数層繰り返す
        output = self.transformer(embedded, mask=mask)
        
        # 6. 全結合層で各文字の予測スコアに変換
        # [batch_size, seq_len, embedding_dim] → [batch_size, seq_len, vocab_size]
        output = self.fc(output)
        
        return output


def find_txt_files_recursively(root_dir):
    """再帰的に全てのtxtファイルを検索"""
    txt_files = []
    
    def scan_directory(path):
        try:
            with os.scandir(path) as entries:
                for entry in entries:
                    try:
                        if entry.is_dir(follow_symlinks=False):
                            scan_directory(entry.path)
                        elif entry.is_file(follow_symlinks=False) and entry.name.endswith('.txt'):
                            txt_files.append(entry.path)
                    except (PermissionError, OSError):
                        continue
        except (PermissionError, OSError):
            pass
    
    scan_directory(root_dir)
    return txt_files


def load_novels(novel_dir=NOVEL_DIR, max_files=None, max_chars=10000000):
    """novelsフォルダから全てのテキストファイルを再帰的に読み込む"""
    text = ""
    total_chars = 0
    files_loaded = 0
    
    print("ファイルを検索中...")
    txt_files = find_txt_files_recursively(novel_dir)
    
    if not txt_files:
        print(f"エラー: {novel_dir}フォルダにテキストファイルが見つかりません")
        return None
    
    print(f"検出されたファイル数: {len(txt_files):,}個")
    if max_files:
        print(f"読み込み上限: {max_files}ファイル")
    print(f"最大文字数: {max_chars:,}文字")
    
    for txt_file in tqdm(txt_files, desc="ファイル読み込み中"):
        if max_files and files_loaded >= max_files:
            print(f"\n上限ファイル数({max_files})に達したため読み込みを終了しました")
            break
        
        if total_chars >= max_chars:
            print(f"\n最大文字数({max_chars:,}文字)に達したため読み込みを終了しました")
            break
        
        file_text = ""
        encodings = ['shift_jis', 'utf-8', 'cp932', 'euc-jp']
        
        for encoding in encodings:
            try:
                with open(txt_file, 'r', encoding=encoding) as f:
                    content = f.read()
                
                # ルビ記号を除去
                content = re.sub(r'《[^》]*》', '', content)
                content = re.sub(r'［[^］]*］', '', content)
                content = re.sub(r'｜', '', content)
                
                # 青空文庫のヘッダー・フッターを除去
                lines = content.split('\n')
                start_idx = 0
                end_idx = len(lines)
                
                for i, line in enumerate(lines):
                    if ('-------' in line or '------' in line or 
                        line.strip() == '' and i < 50):
                        continue
                    if (line.strip() != '' and not line.startswith('http') and
                        '青空文庫' not in line and i < 100):
                        start_idx = i
                        break
                
                for i in range(len(lines)-1, -1, -1):
                    if ('底本' in lines[i] or '入力' in lines[i] or
                        '校正' in lines[i] or '青空文庫' in lines[i]):
                        end_idx = i
                        break
                
                file_text = '\n'.join(lines[start_idx:end_idx]).strip()
                
                if len(file_text) > 100:
                    text += file_text + '\n'
                    total_chars += len(file_text)
                    files_loaded += 1
                    break
            
            except Exception:
                continue
    
    print(f"\n読み込み完了: {files_loaded:,}ファイル, 総文字数: {total_chars:,}文字")
    return text


def create_vocabulary(text):
    """テキストから語彙（使用されている全文字のリスト）を作成"""
    chars = sorted(list(set(text)))
    vocab = {'<PAD>': 0, '<UNK>': 1}
    for i, char in enumerate(chars):
        vocab[char] = i + 2
    idx2char = {v: k for k, v in vocab.items()}
    return vocab, idx2char


def train_model(model, dataloader, criterion, optimizer, scheduler, device, epochs=EPOCHS):
    """
    モデルの学習を実行
    
    【学習の仕組み】
    1. データを読み込む（入力文字列と正解の次の文字列）
    2. モデルで予測を行う（Transformerは全文字を並列処理）
    3. 予測と正解の差（損失）を計算
    4. 誤差逆伝播で勾配を計算
    5. 重みを更新（モデルを改善）
    6. 上記を全データに対して繰り返す（1エポック）
    7. エポック数だけ繰り返す
    
    【LSTMとの違い】
    - LSTMは順次処理だが、Transformerは並列処理
    - そのため、GPUを使った場合の学習速度が大幅に向上
    
    Args:
        model (nn.Module): 学習対象のニューラルネットワークモデル
        dataloader (DataLoader): 学習データを供給するローダー
        criterion: 損失関数（予測と正解の差を計算）
        optimizer: 最適化アルゴリズム（重みの更新方法）
        scheduler: 学習率スケジューラ（学習率を動的に調整）
        device: 計算デバイス（CPU or GPU）
        epochs (int): 学習を繰り返す回数
    """
    
    model.train()
    print(f"\n学習開始（エポック数: {epochs}）")
    
#    best_loss = float('inf')
#    patience = 3
#    patience_counter = 0
    
    # 各エポックの処理
    for epoch in range(epochs):
        total_loss = 0  # このエポックでの累積損失
        # プログレスバーを表示しながらバッチ処理
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}")
        
        for inputs, targets in progress_bar:
            # データをGPU/CPUに転送
            inputs, targets = inputs.to(device), targets.to(device)
            # 勾配をリセット（前のバッチの影響を消す）
            optimizer.zero_grad()

            
            # 順伝播：予測を行う
            # Transformerは全文字を同時に処理（並列化）
            outputs = model(inputs)
            
            # 損失を計算
            # view()で形状を変換：(バッチ, 文字列長, 語彙サイズ) → (バッチ×文字列長, 語彙サイズ)
            loss = criterion(outputs.view(-1, outputs.size(-1)), targets.view(-1))
            
            # 逆伝播：勾配を計算
            loss.backward()
            
            # 勾配クリッピング：勾配が大きくなりすぎるのを防ぐ
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            
            # 重みを更新
            optimizer.step()
            
            total_loss += loss.item()
            progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1}/{epochs}, 平均Loss: {avg_loss:.4f}")
        
        scheduler.step(avg_loss)
        
        # # Early Stopping判定
        # if avg_loss < best_loss - 0.01:  # 0.01以上改善した場合
        #     best_loss = avg_loss
        #     patience_counter = 0
        #     #ベストモデルとして保存
        #     torch.save(model.state_dict(), BEST_MODEL_PATH)
            
        #     print(f"  → ベストモデル更新（Loss: {best_loss:.4f}）")
        # else:
        #     patience_counter += 1
        #     print(f"  → 改善なし（{patience_counter}/{patience}）")
        
        # if patience_counter >= patience:
        #     print(f"\n早期終了: {patience}エポック連続で改善が見られませんでした")
        #     model.load_state_dict(torch.load(BEST_MODEL_PATH, weights_only=True))
            
        #     break





def generate_response(model, prompt, vocab, idx2char, device, max_length=200, temperature=0.8):

    """
    質問（プロンプト）に対する応答を生成
    
    【文章生成の仕組み】
    1. プロンプト（質問文）を入力
    2. Transformerが次の文字を予測（全文字を同時に見て判断）
    3. 予測された文字を入力に追加
    4. 再度次の文字を予測
    5. 2-4を繰り返して文章を生成
    
    【temperature（温度）パラメータ】
    - 低い（0.5など）: 確実な予測を選ぶ → 安定した文章だが単調
    - 高い（1.0以上）: ランダム性が高い → 多様だが不自然な場合も
    
    Args:
        model (nn.Module): 学習済みモデル
        prompt (str): 質問文（文章生成の起点）
        vocab (dict): 文字→ID変換辞書
        idx2char (dict): ID→文字変換辞書
        device: 計算デバイス
        max_length (int): 生成する最大文字数
        temperature (float): 生成のランダム性（0.1〜2.0程度）
    
    Returns:
        str: プロンプトから生成された文章
    """

    
    # モデルを評価モードに設定（Dropoutなどを無効化）
    model.eval()
    

    # プロンプトを数値IDのリストに変換
    input_ids = [vocab.get(c, vocab['<UNK>']) for c in prompt]

    # 最初から長すぎるプロンプトの場合は、末尾 MAX_LENGTH だけ残す
    if len(input_ids) > MAX_LENGTH:
        input_ids = input_ids[-MAX_LENGTH:]


    # 生成テキスト（最初はプロンプトからスタート）
    generated = prompt

    # 勾配計算を無効化（推論時は不要なので高速化）
    with torch.no_grad():
        # 指定された長さまで文字を生成
        for _ in range(max_length):
            # 現在の入力から次の文字を予測。Transformerは毎回全系列を入力する
            #input_tensor = torch.tensor([input_ids]).to(device)
            
            input_tensor = torch.tensor([input_ids], dtype=torch.long).to(device)
            
            output = model(input_tensor)

            # 最後の時刻の出力から次の文字の確率分布を取得
            logits = output[0, -1, :] / temperature  # temperatureで確率分布を調整
            probs = torch.softmax(logits, dim=0)     # ソフトマックスで確率に変換

            
            # 確率分布に従って次の文字をサンプリング （確率が高い文字ほど選ばれやすいが、低確率の文字も選ばれる可能性あり）
            next_char_idx = torch.multinomial(probs, 1).item()
            next_char = idx2char.get(next_char_idx, '')

            # 終了条件：句点などで文が終わり、かつ十分な長さがある場合
            if next_char in ['。', '！', '？', '\n'] and len(generated) > len(prompt) + 10:
                generated += next_char
                break

            # 生成された文字を追加
            generated += next_char
            input_ids.append(next_char_idx)

            # MAX_LENGTHを超えないように古い部分を削除
            if len(input_ids) > MAX_LENGTH:
                input_ids = input_ids[-MAX_LENGTH:]

    return generated



# 計測関数
def _t(msg, t0, device=None):
    if device is not None and device.type == "cuda":
        torch.cuda.synchronize()
    print(f"[TIME] {msg}: {time.perf_counter() - t0:.2f}s") 


def main():
    """
    メイン処理
    
    【プログラムの流れ】
    1. GPUが使用可能かチェック
    2. 学習済みモデルがあるか確認
       - ある場合：モデルを読み込む
       - ない場合：小説を読み込み、学習を実行、モデルを保存
    3. 質問に対して応答を生成・表示
    """
    
    # ============================================================
    # 1. デバイスの設定（GPU or CPU）
    # ============================================================
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用デバイス: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    # ============================================================
    # 2. 学習済みモデルの確認と読み込み or 新規学習
    # ============================================================
    #保存済みのモデルがある場合
    if os.path.exists(MODEL_PATH) and os.path.exists(VOCAB_PATH):
        print("\n学習済みモデルを読み込みます...")
        with open(VOCAB_PATH, 'rb') as f:
            vocab, idx2char = pickle.load(f)
        
        model = NovelTransformer(
            vocab_size=len(vocab),
            embedding_dim=EMBEDDING_DIM,
            num_heads=NUM_HEADS,
            num_layers=NUM_LAYERS,
            ff_dim=FF_DIM
        ).to(device)
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device, weights_only=True))
        print("モデルの読み込みが完了しました")
    
    #保存済みのモデルがない場合　⇒学習
    else:
        print("\n学習済みモデルが見つかりません。新規学習を開始します...")
        
        #時間計測
        t0 = time.perf_counter()
        
        # 小説ファイルの読み込み
        # max_files: 読み込むファイル数の上限   max_chars: 読み込む文字数の上限
        text = load_novels(max_files=3000, max_chars=10_000_000)

        #時間計測
        _t("load_novels", t0)

        if text is None or len(text) < 1000:
            print("エラー: テキストデータが不足しています")
            return
        
        
        print(f"総文字数: {len(text):,}文字")

        #時間計測        
        t0 = time.perf_counter()
        # 語彙の作成（テキスト中の全ユニーク文字を抽出）
        vocab, idx2char = create_vocabulary(text)
        print(f"語彙サイズ: {len(vocab)}")
        
        
        #時間計測
        _t("create_vocabulary", t0)
       
        
        
        # データセットとモデルの準備
        dataset = NovelDataset(text, vocab)
        
        dataloader = DataLoader(
            dataset,
            batch_size=BATCH_SIZE,
            shuffle=True,
            num_workers=4,        # CPU のコア数に合わせて調整
            pin_memory=True,      # GPU への転送を高速化
            persistent_workers=True   # 何度もプロセスを立て直さない
        )        
        
       
        
        print(f"学習サンプル数: {len(dataset):,}")
        
        # モデルの初期化
        model = NovelTransformer(
            vocab_size=len(vocab),
            embedding_dim=EMBEDDING_DIM,
            num_heads=NUM_HEADS,
            num_layers=NUM_LAYERS,
            ff_dim=FF_DIM
        ).to(device)
        
        # パラメータ数の表示
        total_params = sum(p.numel() for p in model.parameters())
        print(f"総パラメータ数: {total_params:,}個")
        
        # 損失関数：予測と正解の差を計算（クロスエントロピー損失）
        criterion = nn.CrossEntropyLoss()
        
        # 最適化アルゴリズム：AdamW（Transformerで推奨されるAdam改良版）
        optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.01)
        
        # 学習率スケジューラを追加
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=2
        )
        

        t0 = time.perf_counter()  # ★追加
        _t("train_model (start sync)", t0, device=device)  # ★追加（開始前に同期）
        
        # 学習の実行
        train_model(model, dataloader, criterion, optimizer, scheduler, device)


        _t("train_model (total)", t0, device=device)  # ★追加（終了後も同期して正しく測る）



        
        # 学習済みモデルの保存
        torch.save(model.state_dict(), MODEL_PATH)
        with open(VOCAB_PATH, 'wb') as f:
            pickle.dump((vocab, idx2char), f)
        print(f"\nモデルを {MODEL_PATH} に保存しました")
    
    # ============================================================
    # 3. チャットボットとして質問に応答
    # ============================================================
    print("\n" + "="*60)
    print("チャットボット起動（Transformer版）")
    print("="*60)
    
    # 質問リスト（ハードコーディング）６点。
    questions = [
        "探偵とは",
        "犯人の心理は",
        "この事件の真相は",
        "不可解な出来事",
        "奇妙な椅子",
        "ゴリラは食べましたか？"
    ]
    
    # 各質問に対して応答を生成
    for question in questions:
        print(f"\n質問: {question}")
        print("-" * 60)
        response = generate_response(model, question, vocab, idx2char, device, 
                                    max_length=300, temperature=0.7)
        print(f"回答: {response}")
        print("-" * 60)


# ============================================================
# プログラムのエントリーポイント
# ============================================================
if __name__ == "__main__":
    if not os.path.exists(NOVEL_DIR):
        print(f"エラー: {NOVEL_DIR} が見つかりません")
        print("パスを確認してください")
    else:
        main()


##############################
# ============================================================
# チャットボット起動（Transformer版）
# ============================================================

# 質問: 探偵とは
# ------------------------------------------------------------
# 回答: 探偵とは、日本人とは皆、糞喰いとなっている。
# ------------------------------------------------------------

# 質問: 犯人の心理は
# ------------------------------------------------------------
# 回答: 犯人の心理は、欧洲大戦以後の闘争に反対している。
# ------------------------------------------------------------

# 質問: この事件の真相は
# ------------------------------------------------------------
# 回答: この事件の真相はすべて深く傷つきそうなものだった。
# ------------------------------------------------------------

# 質問: 不可解な出来事
# ------------------------------------------------------------
# 回答: 不可解な出来事に就いて、夢想していた事であった。
# ------------------------------------------------------------

# 質問: 奇妙な椅子
# ------------------------------------------------------------
# 回答: 奇妙な椅子に腰かけて、いままでは、私にも、この少年少女が、そのような聡明な表情をしているのではないかと思われる。
# ------------------------------------------------------------

# 質問: ゴリラは食べましたか？
# ------------------------------------------------------------
# 回答: ゴリラは食べましたか？
# 　この世の中では、わたしたちは、気がついていました。
# ------------------------------------------------------------
