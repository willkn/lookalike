<template>
  <div class="celebrity-lookalike">
    <h1>Find Your Celebrity Lookalike</h1>
    <div class="camera-container">
      <video ref="video" :width="640" :height="480" class="flipped" autoplay></video>
    </div>
    <button @click="findLookalike" class="find-button">Find Your Celebrity Lookalike</button>
  </div>
</template>

<script>
export default {
  name: 'CelebrityLookalike',
  data() {
    return {
      stream: null,
    }
  },
  mounted() {
    this.startCamera()
  },
  methods: {
    async startCamera() {
      try {
        this.stream = await navigator.mediaDevices.getUserMedia({ video: true })
        this.$refs.video.srcObject = this.stream
      } catch (error) {
        console.error('Error accessing camera:', error)
      }
    },
    findLookalike() {
      // This is where you'll implement the celebrity matching logic
      console.log('Finding your celebrity lookalike...')
    },
  },
  beforeUnmount() {
    if (this.stream) {
      const tracks = this.stream.getTracks()
      tracks.forEach(track => track.stop())
    }
  },
}
</script>

<style scoped>
.celebrity-lookalike {
  display: flex;
  flex-direction: column;
  align-items: center;
  padding: 20px;
}

.camera-container {
  margin-bottom: 20px;
}

.find-button {
  padding: 10px 20px;
  font-size: 16px;
  background-color: #4CAF50;
  color: white;
  border: none;
  border-radius: 5px;
  cursor: pointer;
}

.find-button:hover {
  background-color: #45a049;
}

.flipped {
  transform: scaleX(-1); /* Flips the video element horizontally */
}
</style>