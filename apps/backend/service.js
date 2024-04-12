const express = require('express');
const bodyParser = require('body-parser');
const fs = require('fs');
const path = require('path');
const cors = require('cors'); // 导入 cors
const app = express();
const multer = require('multer');
const upload = multer({ dest: 'uploads/' }); // 设置文件存储路径
const sharp = require('sharp');

const port = 4000;

app.use(cors()); // 允许所有来源的跨域请求

app.use(bodyParser.json({limit: '50mb'})); // 增加body大小限制


app.post('/uploadImage', upload.single('image'), async (req, res) => {
  console.log('Received request for ID:', req.body.id);

  const idStr = String(req.body.id);
  const nameStr = String(req.body.name);
  const originalDir = path.resolve(__dirname, '../classify/original', idStr);
  const processedDir = path.resolve(__dirname, '../classify/processed', idStr);

  console.log('Directories:', { originalDir, processedDir });

  // 确保目录存在
  if (!fs.existsSync(originalDir)) {
    console.log('Creating original directory');
    fs.mkdirSync(originalDir, { recursive: true });
  }
  if (!fs.existsSync(processedDir)) {
    fs.mkdirSync(processedDir, { recursive: true });
  }

  const originalFilePath = path.join(originalDir, `${nameStr}.png`); // 保存原图

  // 移动并重命名上传的文件到原图目录
  fs.rename(req.file.path, originalFilePath, async (err) => {
    if (err) {
      console.error('Error saving the original image:', err);
      return res.status(500).send('Error saving the original image');
    }

    // 随机采样并保存处理后的图像
    try {
      for (let i = 1; i <= 16; i++) {
        // 生成随机裁剪位置
        const left = Math.floor(Math.random() * (500 - 224));
        const top = Math.floor(Math.random() * (500 - 224));
        await sharp(originalFilePath)
          .extract({ width: 224, height: 224, left: left, top: top })
          .toFile(path.join(processedDir, `${nameStr}-${i}.png`));
      }

      res.json({ message: 'Original and processed images saved successfully' });
    } catch (error) {
      console.error('Error processing images:', error);
      res.status(500).send('Error processing images');
    }
  });
});


app.listen(port, () => {
  console.log(`Server running at http://localhost:${port}`);
});
