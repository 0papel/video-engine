// URL de tu servicio en Render (cámbiala si tu subdominio es distinto)
const RENDER_ENDPOINT_URL = "https://video-engine-v1.onrender.com/generar";

/**
 * Procesa las propiedades activas que tengan URLs_Web pero aún no tengan URL_Video.
 */
function generarVideosFaltantes() {
  const sheet = SpreadsheetApp.getActiveSpreadsheet().getSheetByName("CRM_Renewal_Backend");
  if (!sheet) {
    console.error("No se encontró la hoja CRM_Renewal_Backend");
    return;
  }

  const data = sheet.getDataRange().getValues();
  const headers = data[0];

  const idxId = headers.indexOf("ID_Propiedad"); // O la columna de ID que uses
  const idxUrlWeb = headers.indexOf("URLs_Web");
  const idxUrlVideo = headers.indexOf("URL_Video");

  if (idxUrlWeb === -1 || idxUrlVideo === -1) {
    console.error("Faltan las columnas URLs_Web o URL_Video en la planilla");
    return;
  }

  for (let i = 1; i < data.length; i++) {
    const urlsRaw = String(data[i][idxUrlWeb] || "").trim();
    const videoExistente = String(data[i][idxUrlVideo] || "").trim();
    const idProp = idxId !== -1 && data[i][idxId] ? String(data[i][idxId]).trim() : "prop_" + i;

    // Solo procesamos si hay fotos en Cloudinary y todavía no tiene video generado
    if (urlsRaw !== "" && videoExistente === "") {
      const listaFotos = urlsRaw.split(",").map(u => u.trim()).filter(Boolean).slice(0, 5);

      if (listaFotos.length === 0) continue;

      console.log(`Disparando render para fila ${i + 1} (${listaFotos.length} fotos)...`);

      const payload = {
        "id": idProp,
        "fotos": listaFotos
      };

      const options = {
        "method": "post",
        "contentType": "application/json",
        "payload": JSON.stringify(payload),
        "muteHttpExceptions": true
      };

      try {
        const res = UrlFetchApp.fetch(RENDER_ENDPOINT_URL, options);
        const code = res.getResponseCode();
        const json = JSON.parse(res.getContentText());

        if (code === 200 && json.url_video) {
          sheet.getRange(i + 1, idxUrlVideo + 1).setValue(json.url_video);
          console.log(`Fila ${i + 1} completada: ${json.url_video}`);
        } else {
          console.error(`Error en Render para fila ${i + 1}:`, json);
        }
      } catch (err) {
        console.error(`Fallo de conexión en fila ${i + 1}: ${err.toString()}`);
      }
    }
  }
}

/**
 * Función de prueba para forzar una sola fila específica (ej: fila 3)
 */
function probarDisparoFilaEspecifica() {
  const sheet = SpreadsheetApp.getActiveSpreadsheet().getSheetByName("CRM_Renewal_Backend");
  const fila = 3; // Fila de prueba

  const data = sheet.getDataRange().getValues();
  const headers = data[0];
  const idxUrlWeb = headers.indexOf("URLs_Web");
  const idxUrlVideo = headers.indexOf("URL_Video");

  const urlsRaw = String(data[fila - 1][idxUrlWeb] || "").trim();
  const listaFotos = urlsRaw.split(",").map(u => u.trim()).filter(Boolean).slice(0, 5);

  console.log(`Enviando ${listaFotos.length} fotos de la fila ${fila} a Render...`);

  const payload = {
    "id": "prueba_fila_" + fila,
    "fotos": listaFotos
  };

  const options = {
    "method": "post",
    "contentType": "application/json",
    "payload": JSON.stringify(payload),
    "muteHttpExceptions": true
  };

  const res = UrlFetchApp.fetch(RENDER_ENDPOINT_URL, options);
  const json = JSON.parse(res.getContentText());

  if (json.url_video) {
    sheet.getRange(fila, idxUrlVideo + 1).setValue(json.url_video);
    console.log("¡Éxito! Video guardado en columna X: " + json.url_video);
  } else {
    console.error("Fallo:", json);
  }
}
