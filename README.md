## 各區ubike使用與確診數分析

緒論

研究動機環繞在疫情爆發下的雙北地區，在三級警戒下，大眾面臨停課不停工的新生活模式，雖然大眾運輸不再是通勤上班族的首選，但是疫情爆發下我們還是可見許多利用捷運、公車通勤的上班族。因此，我們欲藉由調查雙北各區每日確診人數以及Ubike的每日車流量，來觀察民眾在使用ubike的時候到是有助於防疫（負相關）還是其實更惡化疫情（正相關）。

方法論

資料蒐集
我們先將YouBike雙北市公共自行車即時資訊網站進行資料索取，研究日期為6/12~6/16，時間為7:00~24:00，每20分鐘一次，爬取每一個車站當下的空車位數。

資料整理
對於原始資料的調整，我們先將時段間的空車位數進行加減，計算雙北Ubike站每天的淨車流變動量，用pandas將資料格式化，並使用xlwings將數據轉換到excel，同時抓取雙北每日確診人數的excel檔資料，最後將這兩份資料進行彙整。

資料分析
利用sklearn套件統計這次的資料，透過簡單線性回歸(LinearRegression)和脊回歸(RidgeCV)，希望可以看到數據表面以外的資訊。資料樣本上面，第一組沒有處理前的資料，取得每日各區的ubike站之敘述統計；第二組資料，將其敘述統計之數字透過ubike站總車位數換算各區使用活躍比例。最後再簡單加上時間因子，舉例 : 0612: 1, 0613: 2, ...。再將資料隨機抽樣進行training和testing組的分類，餵入模型。

分析呈現
第一步劃出疫情熱區 : 利用matplotllib.pyplot去製圖，首先先取得雙北各區的地區外殼圖(shapefile)，在透過Geopandas去劃出各區域界線，合併前面整理出來的各區確診數當作判斷依據，將數據分成五等分透過顏色表達嚴重程度；再來將各ubike站的使用前百分之五的站點，以點狀散步圖疊圖到疫情熱區，並以點點半徑表示人次多寡。

研究結果

根據簡單回歸和脊回歸的結果，第一組資料中，在training 裡面我們只分別得到23%和13%的R square，第二組資料，皆得到約21-22%的R square。在模型選擇上面當然是不盡理想的。透過不同模型和不同屬性的資料，我們都沒辦法看到使用人次的四分位數對應到的係數有明顯的趨勢或相關性，但是在各區使用人次標準差上面卻看到較高的正相關；在疫情過去越多天時間係數呈現較高的負相關。

在testing裡面，在R square表現都非常差，約莫落在-4%和-14%之間，也就代表這個模型在預測能力上面極差，這些數據跟確診數統計上沒有關聯，簡單檢討的話，分析交通運輸流量，捷運、公車的旅運量或許是更重要的參考依據，但由於資料取得無法及時跟上，我們並沒有將這些資料納入考量，未來如果加入這段時間的捷運、公車等相關資料，我們的研究結果將會更有機會解釋疫情與交通工具使用人數的關連性。

![image](https://github.com/AmazingGraceChiChi/Ubike-and-Covid-19-analysis/blob/main/images/image1.png)
![image](https://github.com/AmazingGraceChiChi/Ubike-and-Covid-19-analysis/blob/main/images/image5.png)

資料視覺化
附圖1~6

研究討論/結論

統計結果其實並非特別意料之外，畢竟模型之中就是需要捕捉有顯著的資料，我們猜測各區的交通狀況只是其中的一環，並竟透過ubike通勤的民眾是人口中極少數，且資料指選取疫情三級下的五天，加上確診的民眾也不是隨機散佈在各年齡層。在簡單的統計分析之後，除了ubike各區使用次數的標準差有較其他變數有較高的正相關，我們看不到ubike對於各區疫情確診數呈現特別趨勢或相關性。解釋上，我們猜測因為各區的標準差會受到各區極端值的影響，因此在各區若是有交通熱點成為統計上的極端值，其在疫情確診數上面才有比較明顯的趨勢。


附圖1
<img src="https://github.com/AmazingGraceChiChi/Ubike-and-Covid-19-analysis/blob/main/images/image4.png" width=30% height=30%>

附圖2~6
<img src="https://github.com/AmazingGraceChiChi/Ubike-and-Covid-19-analysis/blob/main/images/image2.png" width=30% height=30%>
<img src="https://github.com/AmazingGraceChiChi/Ubike-and-Covid-19-analysis/blob/main/images/image6.png" width=30% height=30%> <img src="https://github.com/AmazingGraceChiChi/Ubike-and-Covid-19-analysis/blob/main/images/image7.png" width=30% height=30%>

<img src="https://github.com/AmazingGraceChiChi/Ubike-and-Covid-19-analysis/blob/main/images/image8.png" width=30% height=30%> <img src="https://github.com/AmazingGraceChiChi/Ubike-and-Covid-19-analysis/blob/main/images/image9.png" width=30% height=30%>

資料來源
雙北市政府資料開放平台
台北市：
https://tcgbusfs.blob.core.windows.net/blobyoubike/YouBikeTP.json
<br>
新北市：https://data.ntpc.gov.tw/api/datasets/71CD1490-A2DF-4198-BEF1-318479775E8A/json?page=0&size=800

台灣各區地域資料
https://data.gov.tw/dataset/7441
雙北地區病例數
https://covid-19.nchc.org.tw/city_confirmed.php



