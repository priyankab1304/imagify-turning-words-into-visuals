document.getElementById("imageForm").onsubmit = async function(event) {
    event.preventDefault();
    let prompt = document.getElementById("prompt").value;
    
    let response = await fetch("/generate-image", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ prompt })
    });
    
    let data = await response.json();
    if (data.success) {
        document.getElementById("generatedImage").src = data.image_url;
        document.getElementById("generatedImage").style.display = "block";
    } else {
        alert("Error: " + data.error);
    }
};
