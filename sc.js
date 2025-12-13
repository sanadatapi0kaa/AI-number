//初期設定
console.log("import完了");
const fs = require("fs");
const path = require("path");
const readlineSync = require("readline-sync");
const Jimp = require("jimp");
const { exec } = require('child_process');

const fff = ["weight1", "weight2", "biases1", "biases2"];
let modecons = false;
let auto = false;
let autocount = 1;
let quiet = false;
let learn = true;
let learnbit = false;
let hidelayer = 32;
let autotimes = 74; //auto base now 800 * 1
let batch = 32; //batch
let batcht = 0; //batch_count
let omomig = 0.01;   //omomi_gakusyuuritu
let baiasug = 0.01;  //baiasu_gakusyuuritu
let flatInput = 0;
let hidden = 0;
let output = 0;
let images = [];
let labels = [];
let nowtimes = 0;

// kari hennsuu
let correct = 0;
let times = 0;
let ans1 = 0;
let ans2 = 0;
let ansmax1 = 0;
let ansmax2 = 0;
let badcount = 0;

let weight1 = 0; //in - hide [16[784]]
let weight2 = 0; //hide - out [10[16]]
let biases1 = 0; //hide
let biases2 = 0; //out

let sa1 = 0; //in - hide weight1 [16[784]]
let sa2 = 0; //hide - out weight2 [10[16]]
let sa3 = 0; //hide biases1 16
let sa4 = 0; //out biases2 10
let delta_hidden = 0;

douki(1);

//softmax
function softmax(arr) {
  if (modecons) console.log("softmax");
  const maxVal = Math.max(...arr);
  const expArr = arr.map((x) => Math.exp(x - maxVal));
  const sum = expArr.reduce((a, b) => a + b, 0);
  return expArr.map((x) => x / sum);
}

//ReLu関数
function leakeyrelu(x) {
  return x > 0 ? x : x * 0.01;
}

//He初期化の値を返す
function heInit(inputSize, outputSize) {
  if (modecons) console.log("Heinit");
  const scale = Math.sqrt(2 / inputSize);
  let returnrow = [];
  for (let i = 0; i < outputSize; i++) {
      const row = [];
      for (let j = 0; j < inputSize; j++) {
          row.push((Math.random() * 2 - 1) * scale);
      }
      returnrow.push(row);
  }
  return returnrow;
}

//関数たちをリセット
function resetnum(input) {
if (input === 0 || input === 1) {
  weight1 = heInit(784, hidelayer);
  fs.writeFileSync("weight1.json", JSON.stringify(weight1));
}

if (input === 0 || input === 2) {
  weight2 = heInit(hidelayer, 10);
  fs.writeFileSync("weight2.json", JSON.stringify(weight2));
}

if (input === 0 || input === 3) {
  biases1 = Array.from({ length: hidelayer }, () => 0);
  fs.writeFileSync("biases1.json", JSON.stringify(biases1));
}

if (input === 0 || input === 4) {
  biases2 = Array.from({ length: 10 }, () => 0);
  fs.writeFileSync("biases2.json", JSON.stringify(biases2));
}

if (input === 0) {
  correct = 0;
  times = 0;
  ansmax1 = 0;
  ansmax2 = 0;
  badcount = 0;
  fs.writeFileSync("log2.txt", "");
}

console.log('新装開店！   ' + input);
}

//inputファイルをデータ化
async function processImage(filePath, index) {
  if (modecons) console.log("prosessimage");
  if (learnbit) {
    return [images[index], labels[index]];
  }else {
    const img = await Jimp.read(filePath);
    img.resize(28, 28).grayscale(); // 28x28 にリサイズし、グレースケール化
    let inputArray = Array.from({ length: 28 }, () => Array(28).fill(0));
    for (let y = 0; y < 28; y++) {
      for (let x = 0; x < 28; x++) {
        const pixel = Jimp.intToRGBA(img.getPixelColor(x, y));
        inputArray[y][x] = pixel.r / 255; // モノクロ前提なので赤成分でOK
      }
    }
    return [inputArray, 0];
  }
}

//inputから計算
function forward(input) {
  if (modecons) console.log("forward");
  hidden = Array(hidelayer).fill(0);
  flatInput = input.flat();

  for (let i = 0; i < hidelayer; i++) {
    for (let j = 0; j < 784; j++) {
      hidden[i] += (flatInput[j] * weight1[i][j]);
    }
    hidden[i] += biases1[i];
    hidden[i] = leakeyrelu(hidden[i]);
  }
  let num = 0;
  for (let i = 0; i < hidden.length; i++) {
    num += hidden[i];
  }

  hidden.forEach((value, index) => {
    if (typeof value !== "number" || isNaN(value)) {
      console.log(`arr[${index}] is not a number:`, value);
      value = 0;
    }
  });

  output = Array(10).fill(0);
  for (let i = 0; i < 10; i++) {
    for (let j = 0; j < hidelayer; j++) {
      output[i] += (hidden[j] * weight2[i][j]);
    }
    output[i] += biases2[i];
  }
  //output is big
  output = softmax(output);
  //softmax all 1> >0
}

//saの計算
function backpropagate(label) {
  if (modecons) console.log("backpropagate");
  //宣言系
  let cost = Array(10).fill(0);
  cost = output.map((o, i) => o - (i === label ? 1 : 0));
  sa1 = Array.from({ length: hidelayer }, () => Array(784).fill(0));
  sa2 = Array.from({ length: 10 }, () => Array(hidelayer).fill(0));
  sa3 = Array(hidelayer).fill(0);
  sa4 = Array(10).fill(0);
  delta_hidden = Array(hidelayer).fill(0);

  //cost
  for (let i = 0; i < 10; i++) {
    cost[i] = output[i] - (i === label ? 1 : 0);
    sa4[i] += cost[i];
  }

  for (let i = 0; i < 10; i++) {
    for (let j = 0; j < hidelayer; j++) {
      sa2[i][j] += cost[i] * hidden[j]
    }
  }

  //delta_hidden
  for (let i = 0; i < hidelayer; i++) {
    for (let j = 0; j < 10; j++) {
      delta_hidden[i] += cost[j] * weight2[j][i];
    }
  }
  
  for (let j = 0; j < hidelayer; j++) {
    const reluDeriv = hidden[j] > 0 ? 1 : 0;
    delta_hidden[j] *= reluDeriv;
  }

  for (let i = 0; i < hidelayer; i++) {
    sa3[i] += delta_hidden[i];
  }
  
  for (let i = 0; i < hidelayer; i++) {
    for (let j = 0; j < 784; j++) {
      sa1[i][j] += delta_hidden[i] * flatInput[j];
    }
  }

  //確認
  cost.forEach((value, index) => {
    if (typeof value !== "number" || isNaN(value)) {
      console.log(`cost[${index}] is not a number:`);
      value = 0;
    }
  });

}

//saの適用
function kaisei(){
  if (modecons) console.log("kaisei");
  for (let i = 0; i < hidelayer; i++) {
    for (let j = 0; j < 10; j++) {
      weight2[j][i] -= sa2[j][i] * omomig * 2;
    }

    biases2[i] -= sa4[i] * baiasug * 2;
  }

  weight2 = weight2.map((row) => row.map((v) => Math.max(-1, Math.min(v, 1))));
  biases2 = biases2.map((v) => Math.max(-1, Math.min(v ?? 0, 1)));

  for (let i = 0; i < hidelayer; i++) {
    for (let j = 0; j < 784; j++) {
      weight1[i][j] -= sa1[i][j] * omomig * 2;
    }
    biases1[i] -= sa3[i] * baiasug * 2;
  }

  weight1 = weight1.map((row) => row.map((v) => Math.max(-1, Math.min(v, 1))));
  biases1 = biases1.map((v) => Math.max(-1, Math.min(v ?? 0, 1)));
}

//ファイルと関数を同期する、なければ作成
function douki(isfirst) {
  if (modecons) console.log("douki");
  let allExist = true;
  fff.forEach((value) => {
    if (!fs.existsSync(`${value}.json`)) {
      fs.writeFileSync(`${value}.json`, "");
      allExist = false;
    }
  });

  if (allExist) {
    weight1 = JSON.parse(fs.readFileSync("weight1.json", "utf8"));
    weight2 = JSON.parse(fs.readFileSync("weight2.json", "utf8"));
    if (isfirst) {
      console.log('weight同期完了');
    }
    biases1 = JSON.parse(fs.readFileSync("biases1.json", "utf8"));
    biases2 = JSON.parse(fs.readFileSync("biases2.json", "utf8"));
    if (isfirst) {
      console.log('biases同期完了');
    }
  }
  
  if (fs.readFileSync("weight1.json", "utf8") == 0) resetnum(1);
  if (fs.readFileSync("weight2.json", "utf8") == 0) resetnum(2);
  if (fs.readFileSync("biases1.json", "utf8") == 0) resetnum(3);
  if (fs.readFileSync("biases2.json", "utf8") == 0) resetnum(4);

  if (isfirst) {
      const path1 = "train-images.idx3-ubyte";
      const buffer1 = fs.readFileSync(path1);
      const numImages1 = buffer1.readUInt32BE(4);
      const numRows = buffer1.readUInt32BE(8);
      const numCols = buffer1.readUInt32BE(12);
    
      const path2 = "train-labels.idx1-ubyte";
      const buffer2 = fs.readFileSync(path2);
      const numLabels = buffer2.readUInt32BE(4);
    
      let offset1 = 16;
    
      for (let i = 0; i < numImages1; i++) {
        const img = [];
        for (let j = 0; j < numRows * numCols; j++) {
          img[j] = buffer1[784 * i + j + offset1] / 255;
        }
        images.push(img);
      }
    
      let offset2 = 8;
      for (let i = 0; i < numLabels; i++) {
        labels.push(buffer2[offset2++]);
      }
    }
}

//AIの総括
async function runTrainingLoop(imageFolderPath, auto) {
  if (modecons) console.log("runtrainingroop");

  //画像とりだし
  const files = fs
    .readdirSync(imageFolderPath)
    .filter((f) => f.endsWith(".png"));

  //画像それぞれ
    for (const file of files) {
    const fullPath = path.join(imageFolderPath, file);
    const match = file.match(/-num(\d+)\.png$/);
    const label = parseInt(match[1], 10);
    const [input, bitlabel] = await processImage(fullPath);

    //画像計算
    forward(input);
    //最大index導出
    const predicted = output.findIndex((x) => Math.abs(x - Math.max(...output)) < Number.EPSILON);
    
    //事後処理、console
      if (predicted === label) correct++;
      times ++;
      if (times > 200 && parseFloat(((correct / times) * 100).toFixed(2)) > parseFloat(ansmax1)) {
        ansmax1 = parseFloat(((correct / times) * 100).toFixed(2));
        ansmax2 = times;
      }

      ans2 = ans1;
      ans1 = ((correct / times) * 100).toFixed(2);
      if (!auto || !quiet) {
        if (parseFloat(ans1) < parseFloat(ans2)) {
          if (predicted === label) console.log(`善 ${file} → 予測: ${predicted}, 正解: ${label}, 正答率: ${ans1}%, 試行回数： ${times} bad`);
          else console.log(`悪 ${file} → 予測: ${predicted}, 正解: ${label}, 正答率: ${ans1}%, 試行回数： ${times} bad`);
          badcount++;
        }else {
          if (predicted === label) console.log(`善 ${file} → 予測: ${predicted}, 正解: ${label}, 正答率: ${ans1}%, 試行回数： ${times}`);
          else console.log(`悪 ${file} → 予測: ${predicted}, 正解: ${label}, 正答率: ${ans1}%, 試行回数： ${times}`);
        }
        quiet = false;
      }
      if (times % 10000 === 0) {
        baiasug = baiasug / 2;
        omomig = omomig / 2;
        console.log(`b, bnを0.001減らしました ${times}回 b= ${omomig}`);
      }
    //log書き出し
    const logfile = path.join(__dirname, 'log2.txt');
    fs.appendFileSync(logfile, ans1 + "\n");
    if (parseInt(ans1) - ans1 === 0) {
      const logfile = path.join(__dirname, 'log.txt');
      fs.appendFileSync(logfile, times + " " + ans1 + "\n");
    }

    //学習
    if (learn) {
      batcht ++;
      backpropagate(label);
      if (batcht >= batch) {
        batcht = 0;
        kaisei();
      }
    }
  }

  fs.writeFileSync("weight1.json", JSON.stringify(weight1));
  fs.writeFileSync("weight2.json", JSON.stringify(weight2));
  fs.writeFileSync("biases1.json", JSON.stringify(biases1));
  fs.writeFileSync("biases2.json", JSON.stringify(biases2));
}

//AIの総括: ubyteファイルを使用
async function runTrainingLoopBit(auto) {
  if (modecons) console.log("runtrainingroopbit");
  let isautocount = 60000;
  if (auto) isautocount = autocount * 800;
  nowtimes = nowtimes % 60000;
  for (let i = nowtimes; i < isautocount + nowtimes; i++) {
    const [input, label]= await processImage(auto, i);

    //画像計算
    forward(input);

    //最大index導出
    const predicted = output.findIndex((x) => Math.abs(x - Math.max(...output)) < Number.EPSILON);
    
    //事後処理、console
      if (predicted === label) correct++;
      times ++;
      if (times > 200 && parseFloat(((correct / times) * 100).toFixed(2)) > parseFloat(ansmax1)) {
        ansmax1 = parseFloat(((correct / times) * 100).toFixed(2));
        ansmax2 = times;
      }

      ans2 = ans1;
      ans1 = ((correct / times) * 100).toFixed(2);
      if (!auto || !quiet) {
        if (parseFloat(ans1) < parseFloat(ans2)) {
          if (predicted === label) console.log(`善 No.${i} → 予測: ${predicted}, 正解: ${label}, 正答率: ${ans1}%, 試行回数： ${times} bad`);
          else console.log(`悪 No.${i} → 予測: ${predicted}, 正解: ${label}, 正答率: ${ans1}%, 試行回数： ${times} bad`);
          badcount++;
        }else {
          if (predicted === label) console.log(`善 No.${i} → 予測: ${predicted}, 正解: ${label}, 正答率: ${ans1}%, 試行回数： ${times}`);
          else console.log(`悪 No.${i} → 予測: ${predicted}, 正解: ${label}, 正答率: ${ans1}%, 試行回数： ${times}`);
        }
        quiet = false;
      }
      if (times % 10000 === 0) {
        baiasug = baiasug / 2;
        omomig = omomig / 2;
        console.log(`b, bnを0.001減らしました ${times}回 b= ${omomig}`);
      }
    //log書き出し
    const logfile = path.join(__dirname, 'log2.txt');
    fs.appendFileSync(logfile, ans1 + "\n");
    if (parseInt(ans1) - ans1 === 0) {
      const logfile = path.join(__dirname, 'log.txt');
      fs.appendFileSync(logfile, times + " " + ans1 + "\n");
    }

    //学習
    if (learn) {
      batcht ++;
      backpropagate(label);
      if (batcht >= batch) {
        batcht = 0;
        kaisei();
      }
    }
  }
  nowtimes += isautocount;

  fs.writeFileSync("weight1.json", JSON.stringify(weight1));
  fs.writeFileSync("weight2.json", JSON.stringify(weight2));
  fs.writeFileSync("biases1.json", JSON.stringify(biases1));
  fs.writeFileSync("biases2.json", JSON.stringify(biases2));
}

//ファイル保存
function saveNetworkToFolder() {
  let folderName = readlineSync.question('ファイル名：');
  const dirPath = path.join(__dirname, folderName);
  if (!fs.existsSync(dirPath)) {
    fs.mkdirSync(dirPath);
    console.log(`📁 フォルダ作成: ${folderName}`);
  }

  //a*20, a*25? 推奨
  let hennsuus = {weight1, weight2, biases1, biases2};

  for (const [name, value] of Object.entries(hennsuus)) {
    const filePath = path.join(dirPath, `${name}.json`);
    const content = `${JSON.stringify(value)}`;
    fs.writeFileSync(filePath, content, "utf-8");
    console.log(`✅ 保存完了: ${name}.json`);
  }

  const src = path.join(__dirname, "log2.txt");
  const dst = path.join(dirPath, "log2.txt");
  fs.copyFileSync(src, dst);
  console.log(`✅ 保存完了: log2.txt`);
}

//AI前の総括
async function foldera(on) {
  if (modecons) console.log("foldera");
  if (on) quiet = true;
  if (on) auto = true;
  else auto = false;
  douki();
  if (learnbit) {
    for (let i = 0; i < autotimes; i++) {
      await runTrainingLoopBit(auto);
    }
  } else {
    const files = fs.readdirSync(__dirname);
    const testCluFiles = files.filter((file) => file.startsWith("test_clu"));
    for (let file of testCluFiles) {
      file = __dirname + "/" + file;
      await runTrainingLoop(file, auto);
    }
  }
  if (on && ((correct / times) * 100).toFixed(2) < 12.00) {
    console.log(((correct / times) * 100).toFixed(2) + '%');
    resetnum(0);
    setTimeout(() => {
      foldera(1);
    }, 500);
  }else {
    if (on) console.log(((correct / times) * 100).toFixed(2) + '%');
    console.log(`max: ${ansmax1}%, times: ${ansmax2} badcount: ${badcount}`);
    if (on && autocount < autotimes) {
      quiet = true;
      autocount++;
      douki();
      foldera(0);
      console.log('start again');
    }else{
      autocount = 1;
      if (modecons) {
        exec('.bat', (err, stdout, stderr) => {
          if (err) {
            console.error(err);
            return;
          }
          console.log(stdout);
        });
      }
      start().catch(err => console.error(err));
    }
  }
}

//開始
async function start() {
  if (modecons) console.log("start");
  let folder = readlineSync.question("選択してくださいc/i/a/b/s/u/l: ");
  if (folder === "c") {
    folder = readlineSync.question("ほんとに？y/n: ");
    if (folder === "y") {
      resetnum(0);
    }
    start();
  } else if (folder === "i") {
    folder = readlineSync.question("画像フォルダのパスを入力してください: ");
    learnbit = false;
    douki();
    runTrainingLoop(folder);
  } else if (folder === "a") {
    douki();
    foldera(0);
  }else if (folder === "ca") {
    douki();
    quiet = true;
    foldera(0);
  }else if (/^a\*\d+$/.test(folder)) {
    douki();
    let split = parseInt(folder.split("*")[1]);
    for (let i = 0; i < split; i++) {
      await foldera(0);
    }
    start().catch(err => console.error(err));
  }else if (folder === "b") {
    console.log(`今： ${omomig}, ${baiasug}`);
    folder = readlineSync.question("値: ");
    if (folder !== "") {
      omomig = parseFloat(folder);
      baiasug = parseFloat(folder);
    }
    console.log(`バイアス：${omomig} ${baiasug}`);
    start();
  }else if (folder === 's') {
    saveNetworkToFolder();
    start();
  }else if (folder === 'u') {
    resetnum(0);
    foldera(1);
  }else if (folder === 't') {
    console.log(`now: ${learnbit}`);
    if (learnbit) learnbit = false;
    else learnbit = true;
    console.log(`change: ${learnbit}`);
    start();
  }else if (folder === 'l'){
    console.log(`now: ${learn}`);
    if (learn) learn = false;
    else learn = true;
    console.log(`change: ${learn}`);
    start();
  }else if (folder === 'help') {
    console.log(`c: clear データ全消し\ni: custom カスタムファイル\na: all file 存在するすべての画像を見る\nb: bias 学習率変更 今:${omomig}\ns: save データを同じ階層に保存\nu: auto 800時点で11%以上のデータのみ通過\nt: 学習方法の変更\nl: learn 現在のデータを読み込むときに学習するかしないか`);
  }else {
    console.log('error');
    start();
  }
}

start();
