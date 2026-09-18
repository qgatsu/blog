# 牌画像

出典: [FluffyStuff/riichi-mahjong-tiles](https://github.com/FluffyStuff/riichi-mahjong-tiles) の `Export/Regular`

ライセンス: **CC0 1.0 Universal**（パブリックドメイン。帰属表示は不要だが出典として記載しておく）
<https://github.com/FluffyStuff/riichi-mahjong-tiles/blob/master/LICENSE.md>

40ファイル、各 600×800 の RGBA PNG。再取得は次のとおり。

```bash
base=https://raw.githubusercontent.com/FluffyStuff/riichi-mahjong-tiles/master/Export/Regular
for f in Man1 Man2 Man3 Man4 Man5 Man5-Dora Man6 Man7 Man8 Man9 \
         Pin1 Pin2 Pin3 Pin4 Pin5 Pin5-Dora Pin6 Pin7 Pin8 Pin9 \
         Sou1 Sou2 Sou3 Sou4 Sou5 Sou5-Dora Sou6 Sou7 Sou8 Sou9 \
         Ton Nan Shaa Pei Haku Hatsu Chun Back Blank Front; do
  curl -sS -o "$f.png" "$base/$f.png"
done
```
