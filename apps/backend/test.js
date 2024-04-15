const { execFile } = require('child_process');
const path = require('path');


console.log(process.cwd())

// 使用绝对路径
const cppExecutablePath = '/localfiles/leather/apps/cppinference/run_inference';
const imageToInferPath = '/localfiles/leather/stone2_texture.jpg'; // 使用绝对路径

// 调用C++可执行文件进行推理
execFile(cppExecutablePath, [imageToInferPath], (error, stdout, stderr) => {
  if (error) {
    console.error('Error during inference:', stderr);
    throw error; // 或者你可以选择更优雅地处理错误
  }

  // 处理输出结果
  const result = parseInt(stdout.trim(), 10);
  if (isNaN(result)) {
    console.error('Invalid inference result:', stdout);
  } else {
    console.log('Inference result:', result);
  }
});
