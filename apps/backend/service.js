const express = require('express');
const bodyParser = require('body-parser');
const fs = require('fs');
const path = require('path');
const cors = require('cors'); // 导入 cors
const app = express();
const multer = require('multer');
const upload = multer({ dest: 'uploads/' }); // 设置文件存储路径

const port = 4000;

app.use(cors()); // 允许所有来源的跨域请求

app.use(bodyParser.json({limit: '50mb'})); // 增加body大小限制

app.post('/uploadImage', upload.single('image'), async (req, res) => {
  console.log('Received request for ID:', req.body.id);
  const nameStr = String(req.body.name);
  const originalBaseDir = path.resolve(__dirname, '../original');

  // 确保基础目录存在
  if (!fs.existsSync(originalBaseDir)) {
    fs.mkdirSync(originalBaseDir, { recursive: true });
  }

  // 获取所有目录并计算下一个目录名
  const directories = fs.readdirSync(originalBaseDir).filter(f => !isNaN(parseInt(f)));
  const maxDirNum = directories.length === 0 ? -1 : Math.max(...directories.map(Number));
  let nextDirName = maxDirNum + 1;

  for (let i = 0; i <= maxDirNum; i++) {
    if (!directories.includes(String(i))) {
      nextDirName = i;
      break;
    }
  }

  const originalDir = path.join(originalBaseDir, String(nextDirName));

  // 确保目录存在
  if (!fs.existsSync(originalDir)) {
    console.log('Creating original directory:', originalDir);
    fs.mkdirSync(originalDir, { recursive: true });
  }

  const originalFilePath = path.join(originalDir, `${nameStr}.png`); // 保存原图

  // 移动并重命名上传的文件到原图目录
  fs.rename(req.file.path, originalFilePath, err => {
    if (err) {
      console.error('Error saving the original image:', err);
      return res.status(500).send('Error saving the original image');
    }
    res.json({ message: 'Original image saved successfully', directory: nextDirName });
  });
});

// app.post('/uploadImage', upload.single('image'), async (req, res) => {
//   console.log('Received request for ID:', req.body.id);

//   const idStr = String(req.body.id);
//   const nameStr = String(req.body.name);
//   const originalDir = path.resolve(__dirname, '../original', idStr);

//   // 确保目录存在
//   if (!fs.existsSync(originalDir)) {
//     console.log('Creating original directory');
//     fs.mkdirSync(originalDir, { recursive: true });
//   }

//   const originalFilePath = path.join(originalDir, `${nameStr}.png`); // 保存原图

//   // 移动并重命名上传的文件到原图目录
//   fs.rename(req.file.path, originalFilePath, async (err) => {
//     if (err) {
//       console.error('Error saving the original image:', err);
//       return res.status(500).send('Error saving the original image');
//     }
//     res.json({ message: 'Original image saved successfully' });
//   });
// });




app.listen(port, () => {
  console.log(`Server running at http://localhost:${port}`);
});
